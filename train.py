import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, recall_score, precision_score, precision_recall_curve, confusion_matrix

from models.bilstm import BiLSTM_Model

eltp_df_train = pd.read_csv('extracted_features/eltp_features/eltp_train.csv')
eltp_df_dev = pd.read_csv('extracted_features/eltp_features/eltp_dev.csv')
lfcc_df_train = pd.read_csv('extracted_features/lfcc_features/lfcc_train.csv')
lfcc_df_dev = pd.read_csv('extracted_features/lfcc_features/lfcc_dev.csv')

lfcc_df_train.columns = ['Unnamed: 0', 'SPEAKER_ID', 'AUDIO_FILE_NAME', 'SYSTEM_ID', '-', 'KEY',
       'LFCC_FEATURE_1', 'LFCC_FEATURE_2', 'LFCC_FEATURE_3', 'LFCC_FEATURE_4', 'LFCC_FEATURE_5',
       'LFCC_FEATURE_6', 'LFCC_FEATURE_7', 'LFCC_FEATURE_8', 'LFCC_FEATURE_9', 'LFCC_FEATURE_10',
       'LFCC_FEATURE_11', 'LFCC_FEATURE_12', 'LFCC_FEATURE_13', 'LFCC_FEATURE_14', 'LFCC_FEATURE_15',
       'LFCC_FEATURE_16', 'LFCC_FEATURE_17', 'LFCC_FEATURE_18', 'LFCC_FEATURE_19', 'LFCC_FEATURE_20']
lfcc_df_dev.columns = ['Unnamed: 0', 'SPEAKER_ID', 'AUDIO_FILE_NAME', 'SYSTEM_ID', '-', 'KEY',
       'LFCC_FEATURE_1', 'LFCC_FEATURE_2', 'LFCC_FEATURE_3', 'LFCC_FEATURE_4', 'LFCC_FEATURE_5',
       'LFCC_FEATURE_6', 'LFCC_FEATURE_7', 'LFCC_FEATURE_8', 'LFCC_FEATURE_9', 'LFCC_FEATURE_10',
       'LFCC_FEATURE_11', 'LFCC_FEATURE_12', 'LFCC_FEATURE_13', 'LFCC_FEATURE_14', 'LFCC_FEATURE_15',
       'LFCC_FEATURE_16', 'LFCC_FEATURE_17', 'LFCC_FEATURE_18', 'LFCC_FEATURE_19', 'LFCC_FEATURE_20']
eltp_lfcc_train = eltp_df_train.merge(lfcc_df_train, on='AUDIO_FILE_NAME')
eltp_lfcc_dev = eltp_df_dev.merge(lfcc_df_dev, on='AUDIO_FILE_NAME')

eltp_lfcc_train['target'] = eltp_lfcc_train['KEY_x'].apply(lambda x: 1 if x =='bonafide' else 0)
eltp_lfcc_dev['target'] = eltp_lfcc_dev['KEY_x'].apply(lambda x: 1 if x =='bonafide' else 0)


X_train, y_train = eltp_lfcc_train[['FEATURE_1', 'FEATURE_2', 'FEATURE_3', 'FEATURE_4', 'FEATURE_5',
       'FEATURE_6', 'FEATURE_7', 'FEATURE_8', 'FEATURE_9', 'FEATURE_10',
       'FEATURE_11', 'FEATURE_12', 'FEATURE_13', 'FEATURE_14', 'FEATURE_15',
       'FEATURE_16', 'FEATURE_17', 'FEATURE_18', 'FEATURE_19', 'FEATURE_20',
       'LFCC_FEATURE_1', 'LFCC_FEATURE_2', 'LFCC_FEATURE_3', 'LFCC_FEATURE_4', 'LFCC_FEATURE_5',
       'LFCC_FEATURE_6', 'LFCC_FEATURE_7', 'LFCC_FEATURE_8', 'LFCC_FEATURE_9', 'LFCC_FEATURE_10',
       'LFCC_FEATURE_11', 'LFCC_FEATURE_12', 'LFCC_FEATURE_13', 'LFCC_FEATURE_14', 'LFCC_FEATURE_15',
       'LFCC_FEATURE_16', 'LFCC_FEATURE_17', 'LFCC_FEATURE_18', 'LFCC_FEATURE_19', 'LFCC_FEATURE_20']], eltp_lfcc_train['target']
X_dev, y_dev = eltp_lfcc_dev[['FEATURE_1', 'FEATURE_2', 'FEATURE_3', 'FEATURE_4', 'FEATURE_5',
       'FEATURE_6', 'FEATURE_7', 'FEATURE_8', 'FEATURE_9', 'FEATURE_10',
       'FEATURE_11', 'FEATURE_12', 'FEATURE_13', 'FEATURE_14', 'FEATURE_15',
       'FEATURE_16', 'FEATURE_17', 'FEATURE_18', 'FEATURE_19', 'FEATURE_20',
       'LFCC_FEATURE_1', 'LFCC_FEATURE_2', 'LFCC_FEATURE_3', 'LFCC_FEATURE_4', 'LFCC_FEATURE_5',
       'LFCC_FEATURE_6', 'LFCC_FEATURE_7', 'LFCC_FEATURE_8', 'LFCC_FEATURE_9', 'LFCC_FEATURE_10',
       'LFCC_FEATURE_11', 'LFCC_FEATURE_12', 'LFCC_FEATURE_13', 'LFCC_FEATURE_14', 'LFCC_FEATURE_15',
       'LFCC_FEATURE_16', 'LFCC_FEATURE_17', 'LFCC_FEATURE_18', 'LFCC_FEATURE_19', 'LFCC_FEATURE_20'
                             ]], eltp_lfcc_dev['target']

model = BiLSTM_Model(weight_saved_path="./checkpoints/eltp_lfcc_500/checkpoint")
model.train_model(X_train, y_train, X_dev, y_dev, epochs=500, batch_size=64)