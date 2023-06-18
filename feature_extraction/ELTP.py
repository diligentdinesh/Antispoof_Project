import librosa
import numpy as np
from scipy.io.wavfile import write
from matplotlib import pyplot as plt

def calculation_of_ternary_code(frame, central_index, theta):
    ternary_code = []
    for i, value in enumerate(frame):
        if i != central_index:
            if value >= (frame[central_index] + theta):
                ternary_code.append(1)
            elif value <= (frame[central_index] + theta) and value >= (frame[central_index] - theta):
                ternary_code.append(0)
            elif value <= (frame[central_index] - theta):
                ternary_code.append(-1)
    return ternary_code
        
def calculate_halves(ternary_code):
    pos_halves = []
    neg_halves = []
    for value in ternary_code:
        if value == 1:
            pos_halves.append(1)
        else:
            pos_halves.append(0)
        
        if value == -1:
            neg_halves.append(1)
        else:
            neg_halves.append(0)
    return pos_halves, neg_halves
    

def adaptive_threshold(frame, alpha=0.6):
    frame = np.array(frame)
    return frame.std() * alpha

def ELTP(sound_signal, frame_length=11):
    total_frame = len(sound_signal)//frame_length
    central_index = frame_length//2

    pos_list = list()
    neg_list = list()
    for i in range(total_frame):
        frame = sound_signal[i*11:(i+1)*11]
        theta = adaptive_threshold(frame=frame)
        ternary_code = calculation_of_ternary_code(frame, central_index, theta)
        pos_halves, neg_halves = calculate_halves(ternary_code)
        pos_int = int("".join([str(i )for i in pos_halves]), 2)
        neg_int = int("".join([str(i )for i in neg_halves]), 2)
        
        pos_list.append(pos_int)
        neg_list.append(neg_int)

    plt.figure()
    plt.subplot(1,2,1)
    plt.hist(pos_list, bins=10, density=True)
    plt.title("POS")
    plt.subplot(1,2,2)
    plt.hist(neg_list, bins=10, density=True)
    plt.title("NEG")
    plt.show()
    pos_hist, _ = np.histogram(pos_list, bins=10, density=True)
    neg_hist, _ = np.histogram(neg_list, bins=10, density=True)
    ELTP = np.concatenate((pos_hist, neg_hist))
    return ELTP

if __name__=="__main__":
    filename = librosa.example('nutcracker')
    y, sr = librosa.load(filename)
    write('sound.wav', sr, y)
    eltp_features = ELTP(y)
    print(eltp_features)