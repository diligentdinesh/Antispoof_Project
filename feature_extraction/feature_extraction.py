import librosa
import os
import matplotlib.pyplot as plt 
from Feature_Model import ELTP, ALDP, LFCC
from tqdm import tqdm
import pandas as pd

train_dir = "/media/niranjan/Volume_1/Voice_Spoofing/datasets/LA/ASVspoof2019_LA_train/flac"
eval_dir = "/media/niranjan/Volume_1/Voice_Spoofing/datasets/LA/ASVspoof2019_LA_eval/flac"
dev_dir = "/media/niranjan/Volume_1/Voice_Spoofing/datasets/LA/ASVspoof2019_LA_dev/flac"

train_features_from_datasets = "/media/niranjan/Volume_1/Voice_Spoofing/datasets/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt"
eval_features_from_datasets  = "/media/niranjan/Volume_1/Voice_Spoofing/datasets/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt"
dev_features_from_datasets   = "/media/niranjan/Volume_1/Voice_Spoofing/datasets/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt"

def get_track_list(directory):
    return os.listdir(directory)

def read_track_file(directory, filename):
    file_name = directory + "/" + filename
    y, sr = librosa.load(file_name, sr=16000)
    return y

track_list = get_track_list(dev_dir)


features_of_all_track = []
# i = 1
for track in tqdm(track_list):
    y = read_track_file(dev_dir, track)
    eltp = LFCC.lfcc(y)
    features = [('FEATURE_'+str(i+1), feature) for i,feature in enumerate(eltp)]
    features =dict(features)
    features['AUDIO_FILE_NAME'] = track.split('.')[0]
    features_of_all_track.append(features)
    # i = i + 1
    # if i==5:
    #     break

features_df = pd.DataFrame(features_of_all_track)


train_features =pd.read_csv(dev_features_from_datasets, header=None, delimiter=r"\s+")
train_features.columns = ['SPEAKER_ID', 'AUDIO_FILE_NAME', 'SYSTEM_ID', '-', 'KEY']
result = train_features.merge(features_df, on='AUDIO_FILE_NAME', how='inner')
result.to_csv('../extracted_features/lfcc_dev.csv')