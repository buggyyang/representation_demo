import torch
from torch import nn
from torch.nn import functional as F
from torch.distributions import Normal


class VAE(nn.Module):
    def __init__(self, roll_dims, hidden_dims,
                 z_dims, n_step, k=1500):
        super(VAE, self).__init__()
        self.gru_0 = nn.GRU(roll_dims, hidden_dims,
                            batch_first=True, bidirectional=True)
        self.linear_mu = nn.Linear(hidden_dims * 2, z_dims)
        self.linear_var = nn.Linear(hidden_dims * 2, z_dims)
        self.grucell_0 = nn.GRUCell(z_dims + roll_dims, hidden_dims)
        self.grucell_1 = nn.GRUCell(hidden_dims, hidden_dims)
        self.grucell_2 = nn.GRUCell(hidden_dims, hidden_dims)
        self.linear_out = nn.Linear(hidden_dims, roll_dims)
        self.linear_init = nn.Linear(z_dims, hidden_dims)
        self.n_step = n_step
        self.roll_dims = roll_dims
        self.hidden_dims = hidden_dims
        self.eps = 1
        self.sample = None
        self.iteration = 0
        self.k = torch.FloatTensor([k])

    def _sampling(self, x):
        idx = x.max(1)[1]
        x = torch.zeros_like(x)
        arange = torch.arange(x.size(0)).long()
        if torch.cuda.is_available():
            arange = arange.cuda()
        x[arange, idx] = 1
        return x

    def encode(self, x):
        # self.gru_0.flatten_parameters()
        x = self.gru_0(x)[-1]
        x = x.transpose_(0, 1).contiguous()
        x = x.view(x.size(0), -1)
        mean = self.linear_mu(x)
        stddev = (self.linear_var(x) * 0.5).exp_()
        return Normal(mean, stddev)

    def decode(self, z):
        out = torch.zeros((z.size(0), self.roll_dims))
        out[:, -1] = 1.
        x, hx = [], [None, None, None]
        t = torch.tanh(self.linear_init(z))
        hx[0] = t
        if torch.cuda.is_available():
            out = out.cuda()
        for i in range(self.n_step):
            out = torch.cat([out, z], 1)
            hx[0] = self.grucell_0(
                out, hx[0])
            if i == 0:
                hx[1] = hx[0]
            hx[1] = self.grucell_1(
                hx[0], hx[1])
            if i == 0:
                hx[2] = hx[1]
            hx[2] = self.grucell_2(
                hx[1], hx[2])
            out = F.log_softmax(self.linear_out(hx[2]), 1)
            x.append(out)
            if self.training:
                p = torch.rand(1).item()
                if p < self.eps:
                    out = self.sample[:, i, :]
                else:
                    out = self._sampling(out)
                self.eps = self.k / \
                    (self.k + torch.exp(self.iteration / self.k))
                self.iteration += 1
            else:
                out = self._sampling(out)
        return torch.stack(x, 1)

    def forward(self, x):
        if self.training:
            self.sample = x.clone()
        dis = self.encode(x)
        if self.training:
            z = dis.rsample()
        else:
            z = dis.mean
        return self.decode(z), dis.mean, dis.stddev
