import numpy as np
import IPython.display
import torch
import matplotlib.pyplot as plt
from src.utils import numpy_to_midi


def z_pitch_plot(idx, z_pitch):
    plt.bar(range(z_pitch.shape[1]), z_pitch[idx])
    plt.ylim(-1, 1)
    plt.rcParams['figure.figsize'] = [10, 6]
    if idx == 0:
        plt.savefig('images/dim_change_p.png', bbox_inches = 'tight')
    plt.show()

def z_rhythm_plot(idx, z_rhythm):
    plt.bar(range(z_rhythm.shape[1]), z_rhythm[idx])
    plt.ylim(-5, 5)
    plt.rcParams['figure.figsize'] = [10, 6]
    if idx == 0:
        plt.savefig('images/dim_change_r.png', bbox_inches = 'tight')
    plt.show()
    
def z_pitch_tune(param, dims, z, model):
    ratio = np.asarray([1, -0.46068906784057617, 0.17136303583780924, 0.158604234457016, 0.1399634579817454])
    ratio = torch.from_numpy(ratio).float()
    if torch.cuda.is_available():
        ratio = ratio.cuda()
    tuned = z + 0
    tuned[:, dims] += ratio[:len(dims)]*param
    output = model.decode(tuned)
    output = output.cpu().data.numpy()
    output = numpy_to_midi(np.concatenate(output, 0), display=True)
    IPython.display.display(IPython.display.Audio(output, rate=22050))
    
def z_rhythm_tune(param, dims, z, model):
    ratio = np.asarray([1, -0.5770807266235352, -0.6188344955444336, 0.3938814401626587, 0.28745582699775696])
    ratio = torch.from_numpy(ratio).float()
    if torch.cuda.is_available():
        ratio = ratio.cuda()
    tuned = z + 0
    tuned[:, dims] += ratio[:len(dims)]*param
    output = model.decode(tuned)
    output = output.cpu().data.numpy()
    output = numpy_to_midi(np.concatenate(output, 0), display=True)
    IPython.display.display(IPython.display.Audio(output, rate=22050))
    
def interpolation(p0, p1, t, dims, model):
    omega = torch.tensor(np.pi/2)
    A = torch.sin((1. - t) * omega) / torch.sin(omega)
    B = torch.sin(t * omega) / torch.sin(omega)
    z0 = p0 + 0
    z0[:, dims] *= A
    z1 = p1 + 0
    z1[:, dims] *= B
    z0[:, dims] += z1[:, dims]
    output = model.decode(z0)
    output = output.cpu().data.numpy()
    output = numpy_to_midi(np.concatenate(output, 0), display=True)
    IPython.display.display(IPython.display.Audio(output, rate=22050))