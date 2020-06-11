import time
import tensorflow as tf

from MTransE.Encoding import encode
from tensorflow.keras import Model
from tensorflow.keras import layers


class TransEmbedding(Model):
    def __init__(self, index_range, dimensions):
        super(TransEmbedding, self).__init__()
        self.rembedding = None
        self.eembedding = None
        self.index_range = index_range
        self.dimensions = dimensions

    def build(self, input_shape):
        self.rembedding = layers.Embedding(self.index_range, self.dimensions)
        self.eembedding = layers.Embedding(self.index_range, self.dimensions)

    def call(self, inputs, **kwargs):
        h = tf.reshape(inputs[0], [-1])
        r = tf.reshape(inputs[1], [-1])
        t = tf.reshape(inputs[2], [-1])
        return self.eembedding(h), self.rembedding(r), self.eembedding(t)


def main():
    t0 = time.time()
    print(tf.__version__)
    # tf.debugging.set_log_device_placement(True)
    path = "data/WK3l-15k/en_de/P_en_v6_training.csv"
    # path = "data/WK3l-15k/en_de/test.csv"
    ds_train = tf.data.Dataset.from_tensor_slices(encode(path, "integer")).shuffle(1024).batch(16384)
    # transe_model = TransEmbedding(9, 5)
    transe_model = TransEmbedding(15108, 75)
    optimizer = tf.keras.optimizers.Adam()

    @tf.function
    def train(inputs, loss_state):
        with tf.GradientTape() as tape:
            inputs = (inputs[:, 0], inputs[:, 1], inputs[:, 2])
            h, r, t = transe_model(inputs)
            h /= tf.norm(h)
            t /= tf.norm(t)
            loss = tf.norm(h + r - t)
            grads = tape.gradient(loss, transe_model.trainable_variables)
            optimizer.apply_gradients(zip(grads, transe_model.trainable_variables))
            loss_state.update_state(loss)

    for epoch in range(1, 15):
        epoch_loss_total = tf.keras.metrics.Sum()
        for datum in iter(ds_train):
            train(datum, epoch_loss_total)
        print("Epoch %d loss %.3f" % (epoch, epoch_loss_total.result()))
        print("Time used ", time.time() - t0)


if __name__ == "__main__":
    main()
