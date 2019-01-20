import numpy as np
import torch
import pretty_midi
import scipy.signal
import librosa.display
import matplotlib.pyplot as plt
from src.models.model import VAE


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

def numpy_to_midi(sample_roll, display=False):
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
        librosa.display.specshow(music.get_piano_roll(100)[48:84],
                                 hop_length=1, sr=100, x_axis='time', y_axis='cqt_note',
                                 fmin=pretty_midi.note_number_to_hz(48))
        plt.figure(figsize=(10, 6))
#         plt.savefig('images/tuned.png')
        plt.show()
    return music.synthesize(wave=scipy.signal.square)

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