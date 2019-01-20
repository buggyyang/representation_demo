import torch
import os
import numpy as np
import utils
from models.model import VAE
from torch.utils.data import TensorDataset, DataLoader

z_dims = 128
data_path = '/scratch/ry649/MHVAE/nice_data'

if __name__ == "__main__":
    model = VAE(130, 2048, z_dims, 32)
    model.eval()
    dic = torch.load('../../params/tr_{}.pt'.format(z_dims))
    for name in list(dic.keys()):
        dic[name.replace('module.', '')] = dic.pop(name)
    model.load_state_dict(dic)
    if torch.cuda.is_available():
        model = model.cuda()
    items = os.scandir(data_path)
    n_batch = 0
    dz_sum_pitch = torch.zeros(12, z_dims)
    for item in items:
        if item.name.endswith('.npy'):
            batch = torch.from_numpy(np.load(item.path)).float()
            dataset = TensorDataset(batch)
            loader = DataLoader(dataset, 500, drop_last=True)
            for subbatch in loader:
                minibatch = subbatch[0]
                if torch.cuda.is_available():
                    minibatch = minibatch.cuda()
                z0 = model.encode(minibatch).mean.mean(0)
                if torch.cuda.is_available():
                    minibatch = minibatch.cpu()
                    z0 = z0.cpu()
                for i in range(12):
                    minibatch_moved = utils.move_pitch(minibatch, i + 1)
                    if torch.cuda.is_available():
                        minibatch_moved = minibatch_moved.cuda()
                    z1 = model.encode(minibatch_moved).mean.mean(0)
                    if torch.cuda.is_available():
                        z1 = z1.cpu()
                    dz_sum_pitch[i] += (z1 - z0).data
                n_batch += 1
                print(n_batch)
    dz_sum_pitch /= n_batch
    torch.save(dz_sum_pitch, '../vectors/z_pitch.pt')
