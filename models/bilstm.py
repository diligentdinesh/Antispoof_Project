import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import FalsePositives, FalseNegatives
from tensorflow.keras.layers import Bidirectional, Dropout
from enum import Enum

import numpy as np
import matplotlib.pyplot as plt 
import matplotlib as mpl
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd

class Mode(Enum):
   eval="eval"
   train="train"

def plot_cm(labels, predictions, p=0.5):
    cm = confusion_matrix(labels, predictions > p)
    plt.figure(figsize=(5,5))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title('Confusion matrix @{:.2f}'.format(p))
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')

    print('Legitimate Transactions Detected (True Negatives): ', cm[0][0])
    print('Legitimate Transactions Incorrectly Detected (False Positives): ', cm[0][1])
    print('Fraudulent Transactions Missed (False Negatives): ', cm[1][0])
    print('Fraudulent Transactions Detected (True Positives): ', cm[1][1])
    print('Total Fraudulent Transactions: ', np.sum(cm[1]))

class BiLSTM_Model:
    def __init__(self, weight_saved_path = "./checkpoints/aldp_lfcc/checkpoint", display=False, learning_rate=0.001, beta_1 = 0.999, mode: Mode="train") -> None:
        self.weight_saved_path = weight_saved_path
        self.display = display
        self.METRICS = [
            tf.keras.metrics.TruePositives(name='tp'),
            tf.keras.metrics.FalsePositives(name='fp'),
            tf.keras.metrics.TrueNegatives(name='tn'),
            tf.keras.metrics.FalseNegatives(name='fn'), 
            tf.keras.metrics.BinaryAccuracy(name='accuracy'),
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.AUC(name='auc'),
            tf.keras.metrics.AUC(name='prc', curve='PR'), # precision-recall curve
        ]
        self. early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_prc', 
            verbose=1,
            patience=10,
            mode='max',
            restore_best_weights=True
        )
        self.adam_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=beta_1 )
        self.build_model()
        if mode!="train":
            self.load_model()

    def build_model(self):
        self.model = Sequential([
            Bidirectional(LSTM(units=1024, name="LSTM_1", return_sequences=True, activation='tanh', recurrent_activation='sigmoid'), input_shape=(40,1), merge_mode='concat'),
            BatchNormalization(),
            Dropout(0.5),
            Bidirectional(LSTM(units=1024, name="LSTM_2", return_sequences=True, activation='tanh', recurrent_activation='sigmoid'), merge_mode='concat'),
            BatchNormalization(),
            Dropout(0.5),
            Bidirectional(LSTM(units=1024, name="LSTM_3", return_sequences=True, activation='tanh', recurrent_activation='sigmoid'), merge_mode='concat'),
            BatchNormalization(),
            Dropout(0.5),
            Bidirectional(LSTM(units=512, name="LSTM_4", return_sequences=True, activation='tanh', recurrent_activation='sigmoid'), merge_mode='concat'),
            BatchNormalization(),
            Dropout(0.5),
            Bidirectional(LSTM(units=512, name="LSTM_5", return_sequences=True, activation='tanh', recurrent_activation='sigmoid'), merge_mode='concat'),
            BatchNormalization(),
            Dropout(0.5),
            Bidirectional(LSTM(units=256, name="LSTM_6", return_sequences=True, activation='tanh', recurrent_activation='sigmoid'), merge_mode='concat'),
            BatchNormalization(),
            Dropout(0.5),
            Bidirectional(LSTM(units=256, name="LSTM_7", return_sequences=True, activation='tanh', recurrent_activation='sigmoid'), merge_mode='concat'),
            BatchNormalization(),
            Dropout(0.5),
            Bidirectional(LSTM(units=128, name="LSTM_8", return_sequences=True, activation='tanh', recurrent_activation='sigmoid'), merge_mode='concat'),
            BatchNormalization(),
            Dropout(0.5),
            Bidirectional(LSTM(units=128, name="LSTM_9", return_sequences=True, activation='tanh', recurrent_activation='sigmoid'), merge_mode='concat'),
            BatchNormalization(),
            Dropout(0.5),
            Bidirectional(LSTM(units=64, name="LSTM_10", activation='tanh', recurrent_activation='sigmoid'), merge_mode='concat'),
            BatchNormalization(),
            Dropout(0.5),
            Dense(units=32, activation='relu', name='Fully_Connected'),
            Dense(units=1, activation='sigmoid', name='Classification')
        ])
        self.model.compile(optimizer=self.adam_optimizer, loss=BinaryCrossentropy(), metrics=self.METRICS)

    def calculate_class_weight(self, y_train):
        neg, pos = np.bincount(y_train)
        total = neg + pos
        weight_for_0 = (1 / neg) * (total / 2.0)
        weight_for_1 = (1 / pos) * (total / 2.0)

        self.class_weight = {0: weight_for_0, 1: weight_for_1}

    def save_model(self):
        self.model.save_weights(self.weight_saved_path)
    
    def load_model(self):
        self.model.load_weights(self.weight_saved_path)

    def display_history(self):
        colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        mpl.rcParams['figure.figsize'] = (12, 10)
        metrics = ['loss', 'prc', 'precision', 'recall']
        for n, metric in enumerate(metrics):
            name = metric.replace("_"," ").capitalize()
            plt.subplot(2,2,n+1)
            plt.plot(self.history.epoch, self.history.history[metric], color=colors[0], label='Train')
            plt.plot(self.history.epoch, self.history.history['val_'+metric],
                    color=colors[0], linestyle="--", label='Val')
            plt.xlabel('Epoch')
            plt.ylabel(name)
            if metric == 'loss':
                plt.ylim([0, plt.ylim()[1]])
            elif metric == 'auc':
                plt.ylim([0.8,1])
            else:
                plt.ylim([0,1])
            plt.legend()


    def train_model(self, X_train, y_train, X_dev, y_dev, epcohs=100, batch_size=30):
        self.calculate_class_weight(y_train=y_train)
        self.history = self.model.fit(X_train, y_train, validation_data=(X_dev, y_dev), epochs=epcohs, batch_size=batch_size, callbacks=[self.early_stopping],class_weight=self.class_weight)
        self.save_model()
        if self.display:
            self.display_history()

    def equal_error_rate(self, df):
        def get_far_frr(df, threshold):
            false_positive = df[(df['y_eval'] == 0) & (df['y_pred'] >= threshold)]   
            false_negative = df[(df['y_eval'] == 1) & (df['y_pred'] < threshold)]  
            spoof_sample = df[df['y_eval'] == 0]
            genuine_sample = df[df['y_eval'] == 1]
            far = len(false_positive) / len(spoof_sample)
            frr = len(false_negative) / len(genuine_sample)
            return far, frr
        thresholds = list(df['y_pred'].sort_values())
        far_, frr_ = [], []
        for thresh in thresholds:
            far, frr = get_far_frr(df, threshold=thresh)
            far_.append(far)
            frr_.append(frr)
        df_f = pd.DataFrame({'far':far_, 'frr': frr_})
        if self.display:
            plt.figure()
            plt.plot(df_f['far'])
            plt.plot(df_f['frr'])
            plt.legend(['False Acceptance Rate', 'False Rejection Rate'])
            plt.xlabel('Thresholds')
            plt.ylabel('Error Rate')
            plt.title('')




    def evaluate_model(self, X_eval, y_eval, batch_size=30):
        eval_predictions = self.model.predict(X_eval, batch_size=batch_size)

        model_results = self.model.evaluate(X_eval, y_eval, batch_size=batch_size, verbose=0)
        for name, value in zip(self.model.metrics_names, model_results):
            print(name, ': ', value)
        plot_cm(y_eval, eval_predictions)
        df = pd.DataFrame({'y_eval':y_eval, 'y_pred': eval_predictions[:,0]})

