import torch
import os
import numpy as np
from models.model import VAE
from torch.utils.data import TensorDataset, DataLoader, ConcatDataset

z_dims = 128
data_path = '/scratch/ry649/MHVAE/raw'
name_list = [
    'full.npy', 'second.npy', 'fourth.npy', 'eighth.npy', 'sixteenth.npy'
]
if __name__ == '__main__':
    model = VAE(130, 2048, z_dims, 32)
    model.eval()
    dic = torch.load('../params/tr_{}.pt'.format(z_dims))
    for name in list(dic.keys()):
        dic[name.replace('module.', '')] = dic.pop(name)
    model.load_state_dict(dic)
    if torch.cuda.is_available():
        model = model.cuda()
    full = torch.from_numpy(np.load(os.path.join(data_path,
                                                 name_list[0]))).float()
    full = full.split(64, 0)
    dz_sum = torch.zeros(4, z_dims)
    for i in range(4):
        sample = torch.from_numpy(
            np.load(os.path.join(data_path, name_list[i + 1]))).float()
        sample = sample.split(64, 0)
        n_batch = 0
        for source, diff in zip(full, sample):
            if torch.cuda.is_available():
                source = source.cuda()
                diff = diff.cuda()
            z0 = model.encode(source).mean
            z1 = model.encode(diff).mean
            dz = (z1 - z0).mean(0)
            dz_sum[i] += dz.cpu().data
            n_batch += 1
    dz_sum /= n_batch
    torch.save(dz_sum, '../vectors/z_rhythm.pt')
