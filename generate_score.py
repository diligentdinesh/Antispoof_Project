import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, recall_score, precision_score, precision_recall_curve, confusion_matrix

from models.bilstm import BiLSTM_Model

eltp_df_eval = pd.read_csv('extracted_features/aldp_features/aldp_eval.csv')
lfcc_df_eval = pd.read_csv('extracted_features/lfcc_features/lfcc_eval.csv')

lfcc_df_eval.columns = ['Unnamed: 0', 'SPEAKER_ID', 'AUDIO_FILE_NAME', 'SYSTEM_ID', '-', 'KEY',
'LFCC_FEATURE_1', 'LFCC_FEATURE_2', 'LFCC_FEATURE_3', 'LFCC_FEATURE_4', 'LFCC_FEATURE_5',
'LFCC_FEATURE_6', 'LFCC_FEATURE_7', 'LFCC_FEATURE_8', 'LFCC_FEATURE_9', 'LFCC_FEATURE_10',
'LFCC_FEATURE_11', 'LFCC_FEATURE_12', 'LFCC_FEATURE_13', 'LFCC_FEATURE_14', 'LFCC_FEATURE_15',
'LFCC_FEATURE_16', 'LFCC_FEATURE_17', 'LFCC_FEATURE_18', 'LFCC_FEATURE_19', 'LFCC_FEATURE_20']

eltp_df_eval = eltp_df_eval.merge(lfcc_df_eval, on='AUDIO_FILE_NAME')

eltp_df_eval['target'] = eltp_df_eval['KEY_x'].apply(lambda x: 1 if x =='bonafide' else 0)
X_eval, y_eval = eltp_df_eval[['FEATURE_1', 'FEATURE_2', 'FEATURE_3', 'FEATURE_4', 'FEATURE_5',
'FEATURE_6', 'FEATURE_7', 'FEATURE_8', 'FEATURE_9', 'FEATURE_10',
'FEATURE_11', 'FEATURE_12', 'FEATURE_13', 'FEATURE_14', 'FEATURE_15',
'FEATURE_16', 'FEATURE_17', 'FEATURE_18', 'FEATURE_19', 'FEATURE_20',
'LFCC_FEATURE_1', 'LFCC_FEATURE_2', 'LFCC_FEATURE_3', 'LFCC_FEATURE_4', 'LFCC_FEATURE_5',
'LFCC_FEATURE_6', 'LFCC_FEATURE_7', 'LFCC_FEATURE_8', 'LFCC_FEATURE_9', 'LFCC_FEATURE_10',
'LFCC_FEATURE_11', 'LFCC_FEATURE_12', 'LFCC_FEATURE_13', 'LFCC_FEATURE_14', 'LFCC_FEATURE_15',
'LFCC_FEATURE_16', 'LFCC_FEATURE_17', 'LFCC_FEATURE_18', 'LFCC_FEATURE_19', 'LFCC_FEATURE_20']], eltp_df_eval['target']

model = BiLSTM_Model(weight_saved_path="./checkpoints/aldp_lfcc_500_double_bilstm_2_weighted_binary_cross_entropy_using_Bootstrap_quantum/checkpoint",mode="eval")
# code from classification of code imbalance

model_results = model.generate_score(X_eval, batch_size=64)
# print(eltp_df_eval.head())
print(eltp_df_eval.index)
score_df = pd.DataFrame({'filename': eltp_df_eval['AUDIO_FILE_NAME'], '-': eltp_df_eval['-_x'], 'KEY': eltp_df_eval['KEY_x'], 'scores': model_results[:,0]})

score_df.to_csv('evaluation/scores/cm_dev_bilstm_weighted_quantum_eval.txt', header=False, index=False, sep=" ")

