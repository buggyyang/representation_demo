import torch, os
import IPython.display
import numpy as np
import matplotlib.pyplot as plt
from src.models.model import VAE
from ipywidgets import interactive, fixed, interact_manual
from src.utils import *
from src.interact_func import *

#load z
z_pitch = torch.load('vectors/z_pitch.pt').numpy()
z_rhythm = torch.load('vectors/z_rhythm.pt').numpy()

model = load_old_model('params/tr_128.pt')

global sample
global target
global z
global zt

def render_pitch():
    interactive_plot = interactive(
        z_pitch_plot, idx=(0, 11), z_pitch=fixed(z_pitch))
    output = interactive_plot.children[-1]
    output.layout.height = '400px'
    return interactive_plot

def render_pitch_sort():
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
    plt.tick_params(axis='both', which='major', labelsize=14)
    ax = plt.gca()
    vals = ax.get_yticks()
    ax.set_yticklabels(['{:,.0%}'.format(x) for x in vals])
    plt.xlabel('dimension indices', fontsize=20)
    plt.ylabel(r'$\overline{\Delta_f z^p}$', fontsize=20)
    plt.axvline(1.5, linestyle='--', alpha=0.5)
    plt.savefig('images/pitch_sort.png', bbox_inches = 'tight')
    plt.show()
    
def render_rhythm():
    interactive_plot_rhythm = interactive(
        z_rhythm_plot, idx=(0, 3), z_rhythm=fixed(z_rhythm))
    output_r = interactive_plot_rhythm.children[-1]
    output_r.layout.height = '400px'
    return interactive_plot_rhythm
    
def render_rhythm_sort():
    for i in range(z_rhythm.shape[0]):
        plt.plot(np.sort(np.abs(z_rhythm[i]))[::-1][:10])
    #     print('significant dimensions after {} splits'.format(i + 1),
    #           np.argsort(np.abs(z_rhythm[i]))[::-1][:5])
    #     print('The corresponding value:', z_rhythm[i][np.argsort(
    #         np.abs(z_rhythm[i]))[::-1][:5]])
    plt.xticks(range(10), range(1, 11))
    plt.tick_params(axis='both', which='major', labelsize=14)
    ax = plt.gca()
    vals = ax.get_yticks()
    ax.set_yticklabels(['{:,.0%}'.format(x) for x in vals])
    plt.xlabel('dimension indices', fontsize=20)
    plt.ylabel(r'$\overline{\Delta_f z^r}$', fontsize=20)
    plt.axvline(4.5, linestyle='--', alpha=0.5)
    plt.savefig('images/rhythm_sort.png', bbox_inches = 'tight')
    plt.show()
    # avg = []
    # avg.append((z_rhythm[:, 45] / z_rhythm[:, 117]).sum() / z_rhythm.shape[0])
    # avg.append((z_rhythm[:, 79] / z_rhythm[:, 117]).sum() / z_rhythm.shape[0])
    # avg.append((z_rhythm[:, 1] / z_rhythm[:, 117]).sum() / z_rhythm.shape[0])
    # avg.append((z_rhythm[:, 73] / z_rhythm[:, 117]).sum() / z_rhythm.shape[0])
    # print(avg)
    
def render_samples():
    return interact_manual(
               choose_samples,
               Sample=[('Use examples', "src"), ('Make your own samples', "tgt")])

def choose_samples(Sample=None):
    
    if Sample == "src":
        print("Example samples have loaded!")        
        
    elif Sample == "tgt":
        render_keyboard()
    else:
        print("\x1b[31mERROR\x1b[0m: You have to choose what kind of sample you want! Please try again")

def load_samples():
    global sample
    global target
    try:
        sample = torch.from_numpy(np.load('data/source_data.npy')).float()
        target = torch.from_numpy(np.load('data/target_data.npy')).float()
    except:
        sample = torch.from_numpy(np.load('data/data_right00.npy')).float()
        target = torch.from_numpy(np.load('data/ngx.npy')).float()
        
    if torch.cuda.is_available():
        sample = sample.cuda()
        target = target.cuda()
    
    global z
    global zt
    z = model.encode(sample).mean
    zt = model.encode(target).mean
    test = numpy_to_midi(
        np.concatenate(sample.cpu().data.numpy(), 0), display=True)
    IPython.display.display(IPython.display.Audio(test, rate=22050))
    test_target = numpy_to_midi(np.concatenate(target.cpu().data.numpy(), 0), display=True)
    IPython.display.display(IPython.display.Audio(test_target, rate=22050))
        
    
def render_adjust_pitch():
    return interact_manual(
               z_pitch_tune,
               parameters=(-10, 10),
               dimensions=[('2dims', [127, 96]), ('5dims', [127, 96, 7, 84, 60]), ('2dims-random', [78, 34])],
               z=fixed(z),
               model=fixed(model))
    
def render_adjust_rhythm():
    return interact_manual(
               z_rhythm_tune,
               parameters=(-5, 5),
               dimensions=[('1dim', [117]), ('5dims', [117, 45, 79, 1, 73]), ('5dims-random', [35, 67, 23, 39, 17])],
               z=fixed(z),
               model=fixed(model))

def render_adjust_rest():
    return interact_manual(
               z_rest_tune,
               parameters=(-5, 5),
               dimensions=fixed(np.setdiff1d(range(128), [117, 45, 79, 1, 73, 127, 96])),
               z=fixed(z),
               model=fixed(model))
    
def render_interpolation():
    return interact_manual(
               interpolation,
               p0=fixed(z),
               p1=fixed(zt),
               interpolation_rate=(0., 1.),
               model=fixed(model),
               dimensions=[('all', list(range(z.shape[1]))), 
                     ('rhythm_direction', np.setdiff1d(range(128), [127, 96]).tolist()), 
                     ('pitch_direction', np.setdiff1d(range(128), [117, 45, 79, 1, 73]).tolist()),
                     ('rest_dimensions', np.setdiff1d(range(128), [117, 45, 79, 1, 73, 127, 96]).tolist())])

def render_keyboard():
    return interact_manual(
               gen_keyboard,
               sample_type=[("source sample", "src"), ("target sample", "tgt")],
               base_tone=[('c3', "c3"), ('c4', "c4"), ('c5', "c5")])
