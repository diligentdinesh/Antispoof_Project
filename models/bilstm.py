import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dense
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import FalsePositives, FalseNegatives
from tensorflow.keras.layers import Bidirectional, Dropout
from enum import Enum
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib as mpl
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd

from sklearn.metrics import roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from models.ocsoftmax import OCSoftmaxLayer, OCSoftmaxLoss


import pennylane as qml
n_qubits = 4 #Number of qubits should be the same as number of features, max number = 25
layers = 1  #layers per block (multiple “layers” of StronglyEntanglingLayers per block )
batch_size=32
dev = qml.device("default.qubit.tf", wires=n_qubits) #Run the model in classical CPU
blocks = 1 #Νumber of blocks (AngleEmbedding and StronglyEntanglingLayers is one block )
@tf.function
@qml.qnode(dev, interface="tf", diff_method="backprop")
def qnode(inputs, weights):
    for i in range(blocks):
        qml.templates.AngleEmbedding(inputs, wires=range(n_qubits))
        qml.templates.StronglyEntanglingLayers(weights[i], wires=range(n_qubits)) #STRONGLY ENTANGLING LAYERS
    return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

weights_shape = (blocks, layers, n_qubits, 3) # Uncomment for Strongly entangling layers
tf.keras.backend.set_floatx("float32")
weight_shapes = {"weights": weights_shape}
inputs = tf.constant(np.random.random((batch_size, n_qubits)))

class Mode(Enum):
   eval="eval"
   train="train"

# def eer_custom(y_true, y_pred):
#   fpr, tpr, thresholds = roc_curve(y_true, y_pred)
#   eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
#   return eer

# def Binary_CrossEntropy_Weighted(y_true, y_pred):
#     w_0 = 9
#     w_1 = 1 
#     return - K.mean((w_0 * y_true * tf.math.log(y_pred) + w_1 * (1.0 - y_true) * tf.math.log(1.0 - y_pred)))


def Binary_CrossEntropy_Weighted(y_true, y_pred): 
    w_0 = 0.9
    w_1 = 0.1 
    # https://github.com/tensorflow/tensorflow/blob/v2.3.1/tensorflow/python/keras/backend.py#L4826
    y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
    y_true = tf.cast(y_true, tf.float32)

    term_0 = y_true * K.log(y_pred + K.epsilon()) # Cancels out when target is 0
    term_1 = (1.0 - y_true) * K.log(1.0 - y_pred + K.epsilon())  # Cancels out when target is 1 

    return -K.mean(w_0 * term_0 + w_1 * term_1, axis=1)

def plot_cm(labels, predictions, p=0.5):
    cm = confusion_matrix(labels, predictions > p)
    plt.figure(figsize=(5,5))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title('Confusion matrix @{:.2f}'.format(p))
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    plt.savefig('confusion_matrix.png')

    print('Legitimate Transactions Detected (True Negatives): ', cm[0][0])
    print('Legitimate Transactions Incorrectly Detected (False Positives): ', cm[0][1])
    print('Fraudulent Transactions Missed (False Negatives): ', cm[1][0])
    print('Fraudulent Transactions Detected (True Positives): ', cm[1][1])
    print('Total Fraudulent Transactions: ', np.sum(cm[1]))

class BiLSTM_Model:
    def __init__(self, weight_saved_path = "./checkpoints/eltp_lfcc/checkpoint", display=False, learning_rate=0.0001, beta_1 = 0.999, mode: Mode="train", fine_tune=False) -> None:
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
        self.early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', 
            verbose=1,
            patience=50,
            mode='min',
            restore_best_weights=True,
            start_from_epoch=200
        )
        self.adam_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=beta_1 )
        self.build_model()
        if mode!="train" or fine_tune:
            print("[INFO] Loading Model")
            self.load_model()

    def build_model(self):
        self.model = Sequential([
            Bidirectional(LSTM(units=64, name="LSTM_1", return_sequences=True, activation='tanh', recurrent_activation='sigmoid'), input_shape=(40,1), merge_mode='concat'),
            BatchNormalization(),
            Dropout(0.5),
            Bidirectional(LSTM(units=64, name="LSTM_2", activation='tanh', recurrent_activation='sigmoid'), merge_mode='concat'),
            BatchNormalization(),
            Dropout(0.5),
            
            Dense(units=512, activation='relu', name='Fully_Connected_1'),
            Dense(units=256, activation='relu', name='Fully_Connected_2'),
            Dense(units=128, activation='relu', name='Fully_Connected_3'),
            Dense(units=64, activation='relu', name='Fully_Connected_4'),
            Dense(units=32, activation='relu', name='Fully_Connected_5'),
    #         Dense(n_qubits, activation="linear", name="before_qbit"),
    # #qlayer= tf.keras.layers.Dense(n_qubits, activation="linear", dtype=tf.float32) #Classical
    #         qml.qnn.KerasLayer(qnode, weight_shapes, n_qubits, dtype=tf.float32), 
            # OCSoftmaxLayer(feat_dim=256)
            Dense(units=1, activation='sigmoid',name="Output_layer")
        ])
        self.model.compile(optimizer=self.adam_optimizer, loss=OCSoftmaxLoss(r_real=0.9, r_fake=0.2, alpha=20.0), metrics=self.METRICS)

    def calculate_class_weight(self, y_train):
        neg, pos = np.bincount(y_train)
        total = neg + pos
        # weight_for_0 = (1 / neg) * (total / 2.0)
        # weight_for_1 = (1 / pos) * (total / 2.0)
        weight_for_0 = neg / total
        weight_for_1 = pos / total

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
        plt.savefig('history_curve.png')


    def train_model(self, train, dev, epochs=100, batch_size=30, datagen=False):
        if datagen:
            self.history = self.model.fit(train,  steps_per_epoch=len(train), epochs=epochs, validation_data=dev, validation_steps=len(dev), callbacks=[self.early_stopping])
        else:
            X_train, y_train = train
            X_dev, y_dev = dev
            self.calculate_class_weight(y_train=y_train)
            self.history = self.model.fit(X_train, y_train, validation_data=(X_dev, y_dev), epochs=epochs, batch_size=batch_size, callbacks=[self.early_stopping],class_weight=self.class_weight)
        self.save_model()
        self.display_history()
        hist_df = pd.DataFrame(self.history.history) 
        # save to json:  
        hist_json_file = 'history.json' 
        with open(hist_json_file, mode='w') as f:
            hist_df.to_json(f)

        # or save to csv: 
        hist_csv_file = 'history.csv'
        with open(hist_csv_file, mode='w') as f:
            hist_df.to_csv(f)

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
        diff_error = 10000
        for thresh in tqdm(thresholds):
            far, frr = get_far_frr(df, threshold=thresh)
            far_.append(far)
            frr_.append(frr)
            if np.abs(far - frr) < diff_error:
                diff_error = np.abs(far-frr)
                eer = far
                th = thresh
        df_f = pd.DataFrame({'far':far_, 'frr': frr_})
        df_f.index = thresholds
        print("Equal Error Rate: ", eer, " at thresholds: ", th)
        plt.figure()
        plt.plot(df_f['far'])
        plt.plot(df_f['frr'])
        plt.legend(['False Acceptance Rate', 'False Rejection Rate'])
        plt.xlabel('Thresholds')
        plt.ylabel('Error Rate')
        plt.title('EER Curve')
        plt.savefig('EER_Curve.png')

        plt.figure()
        df_test = 1 - df_f['frr']
        df_test.index = df_f['far']
        df_test
        plt.figure()
        plt.plot(df_test)
        # plt.legend(['False Acceptance Rate', 'False Rejection Rate'])
        plt.xlabel('Thresholds')
        plt.ylabel('Error Rate')
        plt.title('ROC Curve')
        plt.savefig("ROC_curve.png")

    def generate_score(self, X_eval, batch_size=64):
        print("[INFO] Generating scores")
        eval_predictions = self.model.predict(X_eval, batch_size=batch_size)
        return eval_predictions

    def evaluate_model(self, X_eval, y_eval, batch_size=30):
        print("[INFO] Prediction before evaluation")
        eval_predictions = self.model.predict(X_eval, batch_size=batch_size)

        print("[INFO] Evaluation of Model")
        model_results = self.model.evaluate(X_eval, y_eval, batch_size=batch_size, verbose=2)
        for name, value in zip(self.model.metrics_names, model_results):
            print(name, ': ', value)
        plot_cm(y_eval, eval_predictions)
        
        print("[INFO] Equal Error Rate Calculation")
        df = pd.DataFrame({'y_eval':y_eval, 'y_pred': eval_predictions[:,0]})
        self.equal_error_rate(df)

