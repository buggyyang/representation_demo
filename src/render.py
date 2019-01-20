import torch
import IPython.display
import numpy as np
import matplotlib.pyplot as plt
from src.models.model import VAE
from ipywidgets import interactive, fixed, interact_manual
from src.utils import numpy_to_midi, load_old_model
from src.interact_func import *

z_pitch = torch.load('vectors/z_pitch.pt').numpy()
z_rhythm = torch.load('vectors/z_rhythm.pt').numpy()

sample = torch.from_numpy(np.load('data/data_right00.npy')).float()
target = torch.from_numpy(np.load('data/ngx.npy')).float()
if torch.cuda.is_available():
    sample = sample.cuda()
    target = target.cuda()
model = load_old_model('params/tr_128.pt')
z = model.encode(sample).mean
zt = model.encode(target).mean

def render_pitch():
    interactive_plot = interactive(
        z_pitch_plot, idx=(0, 11), z_pitch=fixed(z_pitch))
    output = interactive_plot.children[-1]
    output.layout.height = '400px'
    return interactive_plot

def render_pitch_sort():
#     fig = plt.figure(figsize=(8,5))
    for i in range(z_pitch.shape[0]):
        plt.plot(np.sort(np.abs(z_pitch[i]))[::-1][:5])
    #     print('significant dimensions indices after {} semitone change:'.format(
    #         i + 1),
    #           np.argsort(np.abs(z_pitch[i]))[::-1][:5])
    #     print('The corresponding value:', z_pitch[i][np.argsort(
    #         np.abs(z_pitch[i]))[::-1][:5]])
    # avg = []
    # avg.append((z_pitch[:, 96] / z_pitch[:, 127]).sum() / z_pitch.shape[0])
    # avg.append((z_pitch[:, 7] / z_pitch[:, 127]).sum() / z_pitch.shape[0])
    # avg.append((z_pitch[:, 84] / z_pitch[:, 127]).sum() / z_pitch.shape[0])
    # avg.append((z_pitch[:, 60] / z_pitch[:, 127]).sum() / z_pitch.shape[0])
    # print(avg)
    plt.xticks(range(5), range(1, 6))
#     fig.savefig('images/pitch_sort.png', bbox_inches = 'tight')
    plt.show()
    
def render_rhythm():
    interactive_plot_rhythm = interactive(
        z_rhythm_plot, idx=(0, 3), z_rhythm=fixed(z_rhythm))
    output_r = interactive_plot_rhythm.children[-1]
    output_r.layout.height = '400px'
    return interactive_plot_rhythm
    
def render_rhythm_sort():
#     fig = plt.figure(figsize=(8,5))
    for i in range(z_rhythm.shape[0]):
        plt.plot(np.sort(np.abs(z_rhythm[i]))[::-1][:10])
    #     print('significant dimensions after {} splits'.format(i + 1),
    #           np.argsort(np.abs(z_rhythm[i]))[::-1][:5])
    #     print('The corresponding value:', z_rhythm[i][np.argsort(
    #         np.abs(z_rhythm[i]))[::-1][:5]])
    plt.xticks(range(10), range(1, 11))
#     fig.savefig('images/rhythm_sort.png', bbox_inches = 'tight')
    plt.show()
    # avg = []
    # avg.append((z_rhythm[:, 45] / z_rhythm[:, 117]).sum() / z_rhythm.shape[0])
    # avg.append((z_rhythm[:, 79] / z_rhythm[:, 117]).sum() / z_rhythm.shape[0])
    # avg.append((z_rhythm[:, 1] / z_rhythm[:, 117]).sum() / z_rhythm.shape[0])
    # avg.append((z_rhythm[:, 73] / z_rhythm[:, 117]).sum() / z_rhythm.shape[0])
    # print(avg)
    
def render_samples():
    test = numpy_to_midi(
        np.concatenate(sample.cpu().data.numpy(), 0), display=True)
    test_target = numpy_to_midi(np.concatenate(target.cpu().data.numpy(), 0), display=True)
    IPython.display.display(IPython.display.Audio(test, rate=22050))
    IPython.display.display(IPython.display.Audio(test_target, rate=22050))
    
def render_adjust_pitch():
    return interact_manual(
               z_pitch_tune,
               param=(-10, 10),
               dims=[('2dims', [127, 96]), ('5dims', [127, 96, 7, 84, 60])],
               z=fixed(z),
               model=fixed(model))
    
def render_adjust_rhythm():
    return interact_manual(
               z_rhythm_tune,
               param=(-5, 5),
               dims=[('1dim', [117]), ('5dims', [117, 45, 79, 1, 73])],
               z=fixed(z),
               model=fixed(model))
    
def render_interpolation():
    return interact_manual(
               interpolation,
               p0=fixed(z),
               p1=fixed(zt),
               t=(0., 1.),
               model=fixed(model),
               dims=[('all', list(range(z.shape[1]))), 
                     ('key_dims', [127, 96, 7, 84, 117, 45, 79, 1, 73]), 
                     ('rest_dims', np.setdiff1d(range(128), [127, 96, 7, 84, 117, 45, 79, 1, 73]).tolist())])