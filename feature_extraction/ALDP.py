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

def aldp_threshold_pos(x, threshold):
    if x <= threshold:
        return 1
    else:
        return 0

def aldp_threshold_neg(x, threshold):
    if x >= threshold:
        return 1
    else:
        return 0
def calculate_alpha_parameter(frame):
    center_value = frame[4]
    frame = np.array(frame)
    return np.sum(frame-center_value) / center_value

def calculate_aldp_value(mask_value, alpha):
    sorted_value_asc = sorted(mask_value)
    sorted_value_dsc = sorted_value_asc[::-1]

    ## Processing for positive aldp code
    pos_dict = dict()
    for i, val in enumerate(sorted_value_dsc):
        if i==0:
            pos_dict[val] = 1
        else:
            delta = sorted_value_dsc[i-1] - sorted_value_dsc[i]
            threshold = alpha * sorted_value_dsc[i]
            pos_dict[val] = aldp_threshold_pos(delta, threshold)
    
    pos_code = list()
    for val in mask_value:
        pos_code.append(pos_dict[val])

    ## Processing for negative aldp code
    neg_dict = dict()
    for i, val in enumerate(sorted_value_asc):
        if i==0:
            neg_dict[val] = 1
        else:
            delta = sorted_value_asc[i-1] - sorted_value_asc[i]
            threshold = - alpha * sorted_value_asc[i]
            neg_dict[val] = aldp_threshold_neg(delta, threshold)
    neg_code = list()
    for val in mask_value:
        neg_code.append(neg_dict[val])
    
    return pos_code, neg_code

def ELDP(sound_signal, frame_length=9):
    total_frame = len(sound_signal) // frame_length

    aldp_list_pos = list()
    aldp_list_neg = list()
    masks = [get_krisch_mask(i) for i in range(8)]
    for i in range(total_frame):
        frame = sound_signal[i * frame_length : (i + 1) * frame_length]
        mask_value = calculate_mask_value(frame, masks)
        alpha = calculate_alpha_parameter(frame)
        pos_code, neg_code = calculate_aldp_value(mask_value=mask_value, alpha=alpha)

        aldp_pos = int("".join([str(i) for i in pos_code]), 2)
        aldp_neg = int("".join([str(i) for i in neg_code]), 2)
        aldp_list_pos.append(aldp_pos)
        aldp_list_neg.append(aldp_neg)
    
    plt.figure()
    plt.hist(aldp_list_pos, bins=10, density=True)
    plt.title("ALDP pos")
    plt.show()

    plt.figure()
    plt.hist(aldp_list_neg, bins=10, density=True)
    plt.title("ALDP neg")
    plt.show()

    ALDP_pos, _ = np.histogram(aldp_list_pos, bins=10, density=True)
    ALDP_neg, _ = np.histogram(aldp_list_neg, bins=10, density=True)
    ELDP = np.concatenate((ALDP_pos, ALDP_neg))
    return ELDP


if __name__ == "__main__":
    filename = librosa.example("nutcracker")
    y, sr = librosa.load(filename)
    write("sound.wav", sr, y)
    eltp_features = ELDP(y)
    print(eltp_features)
