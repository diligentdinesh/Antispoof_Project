import tensorflow as tf
import numpy as np
from tensorflow.keras import backend as K

class OCSoftmaxLayer(tf.keras.layers.Layer):
    def __init__(self, feat_dim=2, r_real=0.9, r_fake=0.5, alpha=20.0):
        super(OCSoftmaxLayer, self).__init__()
        self.feat_dim = feat_dim
        self.center = self.add_weight(
            name='center',
            shape=(1, self.feat_dim),
            initializer=tf.initializers.RandomNormal(mean=0.0, stddev=0.25),
            trainable=True
        )

    def call(self, x):
        w = tf.math.l2_normalize(self.center, axis=1)
        x = tf.math.l2_normalize(x, axis=1)

        scores = tf.matmul(x, tf.transpose(w))
        return tf.identity(scores)


class OCSoftmaxLoss(tf.keras.layers.Layer):
    def __init__(self, r_real=0.9, r_fake=0.2, alpha=5.0):
        super(OCSoftmaxLoss, self).__init__()
        self.r_real = r_real
        self.r_fake = r_fake
        self.alpha = alpha
        self.softplus = tf.keras.layers.Activation('softplus')

    def call(self, y_true, y_pred):
        # y_pred[y_true == 0] = self.r_real - y_pred[y_true == 0]
        # y_pred[y_true == 1] = y_pred[y_true == 1] - self.r_fake
        # y_true = tf.cast(y_true, tf.float32)
        # power = ((0.9-0.4 * y_true) - y_pred) * (-1 ** y_true)
        # loss = K.mean(self.softplus(self.alpha * power))
        scores = tf.where(y_true == 0, self.r_real - y_pred, y_pred - self.r_fake)

        loss = tf.reduce_mean(self.softplus(self.alpha * scores))
        return loss
