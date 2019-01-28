import numpy as np
import IPython.display
import torch
import matplotlib.pyplot as plt
from src.utils import numpy_to_midi
from src.interpolation_plot import *


def z_pitch_plot(idx, z_pitch):
    plt.bar(range(z_pitch.shape[1]), z_pitch[idx])
    plt.ylim(-1, 1)
    plt.rcParams['figure.figsize'] = [10, 6]
    plt.show()

def z_rhythm_plot(idx, z_rhythm):
    plt.bar(range(z_rhythm.shape[1]), z_rhythm[idx])
    plt.ylim(-5, 5)
    plt.rcParams['figure.figsize'] = [10, 6]
    plt.show()
    
def z_pitch_tune(parameters, dimensions, z, model):
    param = parameters
    dims = dimensions
    ratio = np.asarray([1, -0.46068906784057617, 0.17136303583780924, 0.158604234457016, 0.1399634579817454])
    ratio = torch.from_numpy(ratio).float()
    if torch.cuda.is_available():
        ratio = ratio.cuda()
    tuned = z + 0
    tuned[:, dims] += ratio[:len(dims)]*(-param)
    output = model.decode(tuned)
    output = output.cpu().data.numpy()
    output = numpy_to_midi(np.concatenate(output, 0), display=True)
    IPython.display.display(IPython.display.Audio(output, rate=22050))
    
def z_rhythm_tune(parameters, dimensions, z, model):
    param = parameters
    dims = dimensions
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
    
def z_rest_tune(parameters, dimensions, z, model):
    param = parameters
    dims = dimensions
    ratio = 2*np.random.rand(128-7) - 1
    ratio = torch.from_numpy(ratio).float()
    if torch.cuda.is_available():
        ratio = ratio.cuda()
    tuned = z + 0
    tuned[:, dims] += ratio[:len(dims)]*param
    output = model.decode(tuned)
    output = output.cpu().data.numpy()
    output = numpy_to_midi(np.concatenate(output, 0), display=True)
    IPython.display.display(IPython.display.Audio(output, rate=22050))
    
def interpolation(p0, p1, interpolation_rate, dimensions, model):
    dims = dimensions
    omega = torch.tensor(np.pi/2)
    t = interpolation_rate
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
    
    out = list()
    for i in torch.arange(0, 1.1, 0.2).float():
        A = torch.sin((1. - i) * omega) / torch.sin(omega)
        B = torch.sin(i * omega) / torch.sin(omega)
        z0 = p0 + 0
        z0[:, dims] *= A
        z1 = p1 + 0
        z1[:, dims] *= B
        z0[:, dims] += z1[:, dims]
        output = model.decode(z0)
        output = list(output.cpu().data.numpy())
        out.extend(output)
    output = numpy_to_midi(np.concatenate(np.array(out), 0), display=True, interpolation=True)
    IPython.display.display(IPython.display.Audio(output, rate=22050))
    
    plot_interpolation(p0, p1, model)