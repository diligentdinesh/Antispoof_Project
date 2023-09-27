import tensorflow as tf

class OCSoftmax(tf.keras.layers.Layer):
    def __init__(self, feat_dim=2, r_real=0.9, r_fake=0.5, alpha=20.0):
        super(OCSoftmax, self).__init__()
        self.feat_dim = feat_dim
        self.r_real = r_real
        self.r_fake = r_fake
        self.alpha = alpha
        self.center = self.add_weight(
            name="center",
            shape=(1, self.feat_dim),
            initializer=tf.keras.initializers.VarianceScaling(scale=0.25),
            trainable=True
        )
        self.softplus = tf.keras.layers.Activation('softplus')

    def call(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        w = tf.linalg.normalize(self.center, ord=2, axis=1)[0]
        x = tf.linalg.normalize(x, ord=2, axis=1)[0]

        scores = tf.matmul(x, tf.transpose(w))
        output_scores = scores

        scores = tf.where(labels == 0, self.r_real - scores, scores - self.r_fake)

        loss = tf.reduce_mean(self.softplus(self.alpha * scores))

        return loss, output_scores

class AMSoftmax(tf.keras.layers.Layer):
    def __init__(self, num_classes, enc_dim, s=20, m=0.9):
        super(AMSoftmax, self).__init__()
        self.enc_dim = enc_dim
        self.num_classes = num_classes
        self.s = s
        self.m = m
        self.centers = self.add_weight(
            name="centers",
            shape=(num_classes, enc_dim),
            initializer=tf.keras.initializers.RandomNormal()
        )

    def call(self, feat, label):
        batch_size = feat.shape[0]
        norms = tf.norm(feat, ord=2, axis=-1, keepdims=True)
        nfeat = feat / norms

        norms_c = tf.norm(self.centers, ord=2, axis=-1, keepdims=True)
        ncenters = self.centers / norms_c
        logits = tf.matmul(nfeat, tf.transpose(ncenters))

        y_onehot = tf.one_hot(label, self.num_classes)
        margin_logits = self.s * (logits - self.m * y_onehot)

        return logits, margin_logits


if __name__ == "__main__":
    batch_size = 32
    feats_dim = 256
    ocsoftmax = OCSoftmax(feats_dim)
    # ocsoftmax.train()
    # ocsoftmax_optimzer = torch.optim.SGD(ocsoftmax.parameters(), lr=0.01)
    ocsoftmax_optimzer = tf.keras.optimizers.SGD(learning_rate=0.01)
    
    
    g1 = tf.random.Generator.from_seed(1, alg='philox')
    feats = g1.normal(shape=[batch_size, feats_dim])
    import numpy as np 
    labels = np.random.randint(2, size=(batch_size,1))
    outputs, score = ocsoftmax(feats, labels)
    print(outputs, score)

    with tf.GradientTape() as tape:
        lfcc_loss, score = ocsoftmax(feats, labels)
    gradients = tape.gradient(lfcc_loss, ocsoftmax.trainable_variables)
    ocsoftmax_optimzer.apply_gradients(zip(gradients, ocsoftmax.trainable_variables))

