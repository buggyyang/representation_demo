import time
import numpy as np
import torch
import pretty_midi
import scipy.signal
import librosa.display
import IPython.display
from IPython.display import clear_output
import matplotlib
import matplotlib.pyplot as plt
from ipywidgets import Layout, Button, Box, VBox, ButtonStyle, interact_manual, ToggleButtons
from src.models.model import VAE

global note_to_num
global note_style
note_style = ToggleButtons(
            options=['Half Note', 'Quarter Note', 'Eighth Note'],
            description='Note style:',
            disabled=False,
            button_style='', # 'success', 'info', 'warning', 'danger' or '',
        #     icons=['check'] * 3
        )
global src_note_list
global tgt_note_list
src_note_list = list()
tgt_note_list = list()

note_to_num = {"C#3": 49, "C#4": 61, "C#5": 73, "D#3": 51, "D#4": 63, "D#5": 75,
                   "F#3": 54, "F#4": 66, "F#5": 78, "G#3": 56, "G#4": 68, "G#5": 80,
                   "A#3": 58, "A#4": 70, "A#5": 82, "C3": 48, "C4": 60, "C5": 72, "C6": 84,
                   "D3": 50, "D4": 62, "D5": 74, "E3": 52, "E4": 64, "E5": 76,
                   "F3": 53, "F4": 65, "F5": 77, "G3": 55, "G4": 67, "G5": 79,
                   "A3": 57, "A4": 69, "A5": 81, "B3": 59, "B4": 71, "B5": 83}


def interpolation(z1, z2, t, dims=None):
    omega = np.pi / 2
    A = torch.sin((1. - t) * omega) / torch.sin(omega)
    B = torch.sin(t * omega) / torch.sin(omega)
    if dims is None:
        return (A * z1 + B * z2)
    z1[dims] *= A
    z1[dims] += B * z2[dims]
    return z1


def ratio_add(base, scale):
    return base + scale * torch.ones(scale.shape)


def move_pitch(roll, n_semitone):
    roll_shape = roll.shape
    if len(roll_shape) > 2:
        roll = torch.cat(tuple(roll), 0)
    idx = np.where(roll[:, :-2] == 1)
    p_diff = idx[1] + n_semitone
    n_idx = (idx[0], p_diff)
    new_roll = torch.zeros_like(roll[:, :-2])
    new_roll[n_idx] = 1
    new_roll = torch.cat((new_roll, roll[:, -2:]), 1)
    return new_roll.reshape(roll_shape)


def split_rhythm(roll, base_unit):
    roll_shape = roll.shape
    if len(roll_shape) > 2:
        roll = torch.cat(tuple(roll), 0)
    prev, n_unit = -1, 0
    for step in range(roll.shape[0]):
        pitch_num = np.where(roll[step] == 1)[0][0]
        if pitch_num < 128:
            prev = 0 + pitch_num
            n_unit = 1
        elif pitch_num == 128:
            n_unit += 1
        if (n_unit > base_unit) and (pitch_num == 128):
            roll[step] = torch.zeros(roll[step].shape)
            roll[step][prev] += 1
            n_unit = 1
    return roll.reshape(roll_shape)

def numpy_to_midi(sample_roll, display=False, interpolation=False):
    music = pretty_midi.PrettyMIDI()
    piano_program = pretty_midi.instrument_name_to_program(
        'Acoustic Grand Piano')
    piano = pretty_midi.Instrument(program=piano_program)
    t = 0
    for i in sample_roll:
        if 'torch' in str(type(i)):
            pitch = int(i.max(0)[1])
        else:
            pitch = int(np.argmax(i))
        if pitch < 128:
            note = pretty_midi.Note(
                velocity=100, pitch=pitch, start=t, end=t + 1 / 16)
            t += 1 / 16
            piano.notes.append(note)
        elif pitch == 128:
            if len(piano.notes) > 0:
                note = piano.notes.pop()
            else:
                p = np.random.randint(60, 72)
                note = pretty_midi.Note(
                    velocity=100,
                    pitch=int(p),
                    start=0,
                    end=t
                )
            note = pretty_midi.Note(
                velocity=100,
                pitch=note.pitch,
                start=note.start,
                end=note.end + 1 / 16)
            piano.notes.append(note)
            t += 1 / 16
        elif pitch == 129:
            t += 1 / 16
    music.instruments.append(piano)
    if display:
        plt.figure(figsize=(30, 10))
        start, end = bound(m=music.get_piano_roll(100))
        ax = plt.gca()
        ax.tick_params(axis = 'both', which = 'major', labelsize = 44)
#         ax.tick_params(axis = 'both', which = 'minor', labelsize = 20)
        if interpolation:
            shape = music.get_piano_roll(100)[start+1:end-1].shape
            if shape[1] == 9600:
                plt.axvspan(0,16, facecolor="#6600cc", alpha=0.5)
                plt.axvspan(16,32, facecolor="#8510b3", alpha=0.5)
                plt.axvspan(32,48, facecolor="#a3209a", alpha=0.5)
                plt.axvspan(48,64, facecolor="#c23082", alpha=0.5)
                plt.axvspan(64,80, facecolor="#e04069", alpha=0.5)
                plt.axvspan(80,96, facecolor="#ff5050", alpha=0.5)
            else:
                plt.axvspan(0,2, facecolor="#6600cc", alpha=0.5)
                plt.axvspan(2,4, facecolor="#8510b3", alpha=0.5)
                plt.axvspan(4,6, facecolor="#a3209a", alpha=0.5)
                plt.axvspan(6,8, facecolor="#c23082", alpha=0.5)
                plt.axvspan(8,10, facecolor="#e04069", alpha=0.5)
                plt.axvspan(10,12, facecolor="#ff5050", alpha=0.5)
            cmap = matplotlib.colors.ListedColormap(['white', "black"])
            librosa.display.specshow(music.get_piano_roll(100)[start:end],
                                     hop_length=1, sr=100, x_axis='time', y_axis='cqt_note',
                                     fmin=pretty_midi.note_number_to_hz(48), cmap=cmap, shading='flat')
        else:
            librosa.display.specshow(music.get_piano_roll(100)[start:end],
                                     hop_length=1, sr=100, x_axis='time', y_axis='cqt_note',
                                     fmin=pretty_midi.note_number_to_hz(48), cmap="inferno", shading='flat')
        plt.xlabel('time(s)', fontsize=48)
        plt.ylabel('pitch', fontsize=48)
        ax.minorticks_off()
        plt.show()
        
    return music.synthesize(wave=scipy.signal.square)

def bound(m=None):
    start = 0
    end = 0
    for line in range(m.shape[0]):
        if 100 in m[line]:
            start = line
            break
    for line in range(m.shape[0]-1, 0, -1):
        if 100 in m[line]:
            end = line
            break
    bound = (end - start)//2
    return start-bound-1, end+bound+1


def load_old_model(param_path):
    model = VAE(130, 2048, 128, 32)
    model.eval()
    dic = torch.load(param_path)
    for name in list(dic.keys()):
        dic[name.replace('module.', '')] = dic.pop(name)
    model.load_state_dict(dic)
    if torch.cuda.is_available():
        model = model.cuda()
    return model

    
def gen_keyboard(sample_type=None, base_tone=None):
    
    if base_tone == "c3":
        num = 3
    elif base_tone == "c4":
        num = 4
    elif base_tone == "c5":
        num = 5
    else:
        raise AssertionError("Make sure you have choose one \x1b[31mbase_tone\x1b[0m!")
    
    if sample_type == "src":
        Type = "src"
    elif sample_type == "tgt":
        Type = "tgt"
    else:
        raise AssertionError("Make sure you have choose one \x1b[31msample type\x1b[0m!")

    print("\x1b[31mNOTE\x1b[0m: You have to enter enough notes in order to have your own samples!")
    
    items_black = [
        Button(description='', layout=Layout(flex='0.8 1 5%', height='100px', min_width='5px'), button_style='danger', style=ButtonStyle(button_color='white')),
        Button(description='C#{}'.format(num), layout=Layout(flex='1 1 auto', height='100px', min_width='10px'), button_style='danger', style=ButtonStyle(button_color='black')),
        Button(description='D#{}'.format(num), layout=Layout(flex='1 1 auto', height='100px', min_width='10px'), button_style='danger', style=ButtonStyle(button_color='black')),
        Button(description='', layout=Layout(flex='0.8 22%', height='100px', min_width='20px'), button_style='danger', style=ButtonStyle(button_color='white')),
        Button(description='F#{}'.format(num), layout=Layout(flex='1 1 auto', height='100px', min_width='10px'), button_style='danger', style=ButtonStyle(button_color='black')),
        Button(description='G#{}'.format(num), layout=Layout(flex='1 1 auto', height='100px', min_width='10px'), button_style='danger', style=ButtonStyle(button_color='black')),
        Button(description='A#{}'.format(num), layout=Layout(flex='1 1 auto', height='100px', min_width='10px'), button_style='danger', style=ButtonStyle(button_color='black')),
        Button(description='', layout=Layout(flex='1 1 60%', height='100px', min_width='32px'), button_style='danger', style=ButtonStyle(button_color='white')),
    ]

    items_white = [
        Button(description='C{}'.format(num), layout=Layout(flex='1 1 10%', height='100px', min_width='10px', border='2px solid'), style=ButtonStyle(button_color='white')),
        Button(description='D{}'.format(num), layout=Layout(flex='1 1 10%', height='100px', min_width='10px', border='2px solid'), style=ButtonStyle(button_color='white')),
        Button(description='E{}'.format(num), layout=Layout(flex='1 1 10%', height='100px', min_width='10px', border='2px solid'), style=ButtonStyle(button_color='white')),
        Button(description='F{}'.format(num), layout=Layout(flex='1 1 10%', height='100px', min_width='10px', border='2px solid'), style=ButtonStyle(button_color='white')),
        Button(description='G{}'.format(num), layout=Layout(flex='1 1 10%', height='100px', min_width='10px', border='2px solid'), style=ButtonStyle(button_color='white')),
        Button(description='A{}'.format(num), layout=Layout(flex='1 1 10%', height='100px', min_width='10px', border='2px solid'), style=ButtonStyle(button_color='white')),
        Button(description='B{}'.format(num), layout=Layout(flex='1 1 10%', height='100px', min_width='10px', border='2px solid'), style=ButtonStyle(button_color='white')),
        Button(description='C{}'.format(num+1), layout=Layout(flex='1 1 10%', height='100px', min_width='10px', border='2px solid'), style=ButtonStyle(button_color='white')),
    ]
    
    items_note = [
        note_style
    ]
    
    item_restart = [
        Button(description="Restart the recording", button_style='warning', layout=Layout(flex='1 1 auto', height='28px', min_width='10px'))
    ]
    
    item_delete = [
        Button(description="Delete the last recording", button_style='danger', layout=Layout(flex='1 1 auto', height='28px', min_width='10px'))

    ]
    
    item_play = [
        Button(description="Play the recording", button_style='success', layout=Layout(flex='1 1 auto', height='28px', min_width='10px'))
    ]
    
    # part where perform button 
    restart = item_restart[0]
    delete = item_delete[0]
    play = item_play[0]
    
    if Type == "src":
        for white in items_white:
            white.on_click(src_clicked)
            
        for black in items_black:
            if "#" in black.description:
                black.on_click(src_clicked)
                
        restart.on_click(src_reset)
        delete.on_click(src_delete)
        play.on_click(src_play)
        
    else:
        for white in items_white:
            white.on_click(tgt_clicked)
        
        for black in items_black:
            if "#" in black.description:
                black.on_click(tgt_clicked)
        
        restart.on_click(tgt_reset)
        delete.on_click(tgt_delete)
        play.on_click(tgt_play)
    
    
    box_layout = Layout(display='flex',
                        flex_flow='row',
                        align_items='stretch',
                        width='40%',
                       grid_gap="")
    box_play = Box(children=item_play, layout=box_layout)
    box_restart = Box(children=item_restart, layout=box_layout)
    box_delete = Box(children=item_delete, layout=box_layout)
    box_black = Box(children=items_black, layout=box_layout)
    box_white = Box(children=items_white, layout=box_layout)
    box_note = Box(children=items_note, layout=box_layout)
    display(VBox([box_play, box_restart, box_delete, box_black, box_white, box_note]))    
    

def src_clicked(b):
    # check the if the list length is maximum, duration of 2 whole notes
    if len(src_note_list) >= 32:
        clear_output()
        print("You have already entered enough notes! Recording has completed! Make sure the other type of sample has also been recorded!")
        np.save("data/source_data.npy", np.array([src_note_list]))
        del src_note_list[:]
        return
    
    # check which button is presses
    else:
        if note_style.value == "Half Note":
            print("\033[1m\033[94m{} - - - -\033[0m\033[0m".format(b.description), end=' ')
        elif note_style.value == "Quarter Note":
            print("\033[1m\033[94m{} - -\033[0m\033[0m".format(b.description), end=' ')
        else:
            print("\033[1m\033[94m{} -\033[0m\033[0m".format(b.description), end=' ')
        
        note = play_note(int(note_to_num[b.description]), note_style)
        src_note_list.extend(note)
#         print("Note {} has been clicked and recorded!".format(b.description))
        
def tgt_clicked(b):
    # check the if the list length is maximum, duration of 2 whole notes
    if len(tgt_note_list) >= 32:
        clear_output()
        print("You have already entered enough notes! Recording has completed! Make sure the other type of sample has also been recorded!")
        np.save("data/target_data.npy", np.array([tgt_note_list]))
        del tgt_note_list[:]
        return
    
    # check which button is presses
    else:
        if note_style.value == "Half Note":
            print("\033[1m\033[94m{} - - - -\033[0m\033[0m".format(b.description), end=' ')
        elif note_style.value == "Quarter Note":
            print("\033[1m\033[94m{} - -\033[0m\033[0m".format(b.description), end=' ')
        else:
            print("\033[1m\033[94m{} -\033[0m\033[0m".format(b.description), end=' ')
        note = play_note(int(note_to_num[b.description]), note_style)
        tgt_note_list.extend(note)
#         print("Note {} has been clicked and recorded!".format(b.description))
        
        
def src_reset(b):
    del src_note_list[:]
    clear_output()
    print("You have restarted the recording for source sample!")

def tgt_reset(b):
    del tgt_note_list[:]
    clear_output()
    print("You have restarted the recording for target sample!")

def src_delete(b):
    try:
        del src_note_list[-1]
        update_plot(sample="src")
    except:
        print("There is no note to delete!")
    
def tgt_delete(b):
    try:
        del tgt_note_list[-1]
        update_plot(sample="tgt")
    except:
        print("There is no note to delete!")

def src_play(b):
    output = numpy_to_midi(np.concatenate(np.array([src_note_list]), 0))
    display(IPython.display.Audio(output, rate=22050, autoplay=True))
    
def tgt_play(b):
    output = numpy_to_midi(np.concatenate(np.array([tgt_note_list]), 0))
    display(IPython.display.Audio(output, rate=22050, autoplay=True))

def num_to_array(number):
    output = np.zeros(130,)
    if number == -1:
        output[128] = 1
    elif number == -2:
        output[129] = 1
    else:
        output[number] = 1
    return np.array(output)


def play_note(number, note_style):
    hold_array = np.zeros(130,)
    hold_array[128] = 1
    note_array = num_to_array(number)
    music_array = [note_array]
    if note_style.value == "Half Note":
        for i in range(7):
            music_array.append(hold_array)
    elif note_style.value == "Quarter Note":
        for i in range(3):
            music_array.append(hold_array)
    else:
        music_array.append(hold_array)
    output = numpy_to_midi(np.concatenate(np.array([music_array]), 0))
#     display(IPython.display.Audio(output, rate=22050, autoplay=True))
    return music_array
    