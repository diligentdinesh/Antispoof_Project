import tensorflow as tf
import numpy as np 
import pandas as pd 

class CustomDataGen(tf.keras.utils.Sequence):
    
    def __init__(self, df, X_col, y_col,
                 batch_size=32,
                 shuffle=True):
        
        self.df = df.copy()
        self.X_col = X_col
        self.y_col = y_col
        self.positive_df = self.df[self.df[y_col]==1]
        self.negative_df = self.df[self.df[y_col]==0]
        
        self.batch_size = batch_size
        self.negative_batch_size = self.batch_size // 2
        self.positive_batch_size = self.batch_size // 2
        self.shuffle = shuffle
        
        self.n = len(self.negative_df)
        self.m = len(self.positive_df)
    
    def on_epoch_end(self):
        if self.shuffle:
            self.negative_df = self.negative_df.sample(frac=1).reset_index(drop=True)
    
    def __getitem__(self, index):
        neg_batch = self.negative_df[index*self.negative_batch_size: (index+1) * self.negative_batch_size]
        pos_batch = self.positive_df.iloc[np.random.randint(low=0, high=self.m, size=self.positive_batch_size)]
        batch = pd.concat([neg_batch, pos_batch])
        return batch[self.X_col], batch[self.y_col]

    
    def __len__(self):
        return self.n // self.negative_batch_size
    
if __name__ == "__main__":
    aldp_df = pd.read_csv('extracted_features/aldp_features/aldp_train.csv')

    aldp_df['target'] = aldp_df['KEY'].apply(lambda x: 1 if x =='bonafide' else 0)
    X_col=['FEATURE_1', 'FEATURE_2', 'FEATURE_3', 'FEATURE_4', 'FEATURE_5', 'FEATURE_6', 'FEATURE_7', 'FEATURE_8', 'FEATURE_9', 'FEATURE_10', 'FEATURE_11', 'FEATURE_12', 'FEATURE_13', 'FEATURE_14', 'FEATURE_15', 'FEATURE_16', 'FEATURE_17', 'FEATURE_18', 'FEATURE_19', 'FEATURE_20']
    datagen = CustomDataGen(aldp_df, X_col=X_col, y_col='target', batch_size=32)
    X, y = datagen[0]
    print(X.shape)
    print(y.value_counts())
    print(y.shape)