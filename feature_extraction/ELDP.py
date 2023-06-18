import librosa
import numpy as np
from scipy.io.wavfile import write
from matplotlib import pyplot as plt
import scipy.stats as ss

def get_krisch_mask(index=0):
    mask = [5, -3, -3, -3, -3, -3, 5, 5]
    for i in range(index):
        a = mask.pop(0)
        mask.append(a)
    mask.insert(4, 0)
    return mask

def calculate_mask_value(frame, masks):
    mask_value = list()
    frame = np.array(frame)
    masks = np.array(masks)
    for mask in masks:
        mask_value.append(np.sum(frame * mask))
    return mask_value[::-1]

def get_rank(mask_value):
    return ss.rankdata(np.abs(mask_value)).astype('int')

def get_code_bit(rank):
    return list(map(lambda x: 1 if x>=6 else 0, rank))

def ELDP(sound_signal, frame_length=9):
    total_frame = len(sound_signal) // frame_length

    ldp_list = list()
    masks = [get_krisch_mask(i) for i in range(8)]
    for i in range(total_frame):
        frame = sound_signal[i * frame_length : (i + 1) * frame_length]
        mask_value = calculate_mask_value(frame, masks)
        rank = get_rank(mask_value)
        code_bit = get_code_bit(rank)
        ldp = int("".join([str(i) for i in code_bit]), 2)
        ldp_list.append(ldp)

    plt.figure()
    plt.hist(ldp_list, bins=10, density=True)
    plt.title("LDP")
    plt.show()
    ELDP, _ = np.histogram(ldp_list, bins=10, density=True)
    return ELDP


if __name__ == "__main__":
    filename = librosa.example("nutcracker")
    y, sr = librosa.load(filename)
    write("sound.wav", sr, y)
    eltp_features = ELDP(y)
    print(eltp_features)
