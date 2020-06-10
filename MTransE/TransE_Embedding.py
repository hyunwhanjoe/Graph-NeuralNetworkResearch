import time
import tensorflow as tf

from MTransE.Encoding import encode
from tensorflow.keras import Model
from tensorflow.keras import layers


class TransEmbedding(Model):
    def __init__(self):
        super(TransEmbedding, self).__init__()
        self.rembedding = None
        self.eembedding = None

    def build(self, input_shape):
        self.rembedding = layers.Embedding(15108, 75)
        self.eembedding = layers.Embedding(15108, 75)
        # self.rembedding = layers.Embedding(9, 5)
        # self.eembedding = layers.Embedding(9, 5)

    def call(self, inputs, **kwargs):
        h = inputs[0]
        r = inputs[1]
        t = inputs[2]
        return self.eembedding(h), self.rembedding(r), self.eembedding(t)


def main():
    t0 = time.time()
    print(tf.__version__)
    path = "data/WK3l-15k/en_de/P_en_v6_training.csv"
    # path = "data/WK3l-15k/en_de/test.csv"
    ds_train = tf.data.Dataset.from_tensor_slices(encode(path, "integer"))
    transe_model = TransEmbedding()
    optimizer = tf.keras.optimizers.SGD()

    @tf.function
    def train(inputs, loss_state):
        with tf.GradientTape() as tape:
            h, r, t = transe_model(inputs)
            h /= tf.norm(h)
            t /= tf.norm(t)
            loss = tf.norm(h + r - t)
            grads = tape.gradient(loss, transe_model.trainable_variables)
            optimizer.apply_gradients(zip(grads, transe_model.trainable_variables))
        loss_state.update_state(loss)

    for epoch in range(1, 20):
        epoch_loss_total = tf.keras.metrics.Sum()
        for datum in iter(ds_train):
            train(datum, epoch_loss_total)
        print("Epoch %d loss %.3f" % (epoch, epoch_loss_total.result()))
    print("Time used ", time.time() - t0)

if __name__ == "__main__":
    main()
