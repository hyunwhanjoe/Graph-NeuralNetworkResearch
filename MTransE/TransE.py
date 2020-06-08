import time
import tensorflow as tf

from MTransE.OneHotEncoding import one_hot_encode
from tensorflow.keras import Model


class TransE(Model):
    def __init__(self):
        super(TransE, self).__init__()
        self.rembedding = None
        self.eembedding = None

    def build(self, input_shape):  # input_shape is needed
        self.rembedding = tf.keras.layers.Dense(75, use_bias=False)  # Normalize relations
        self.eembedding = tf.keras.layers.Dense(75, use_bias=False)

    def call(self, inputs, **kwargs):
        h, r, t = inputs
        return self.eembedding(h), self.rembedding(r), self.eembedding(t)


def main():
    t0 = time.time()
    print(tf.__version__)
    path = "data/WK3l-15k/en_de/test.csv"
    ds_train = tf.data.Dataset.from_tensor_slices(one_hot_encode(path)).batch(1)  # mini-batch?
    transe_model = TransE()
    optimizer = tf.keras.optimizers.SGD()  # ()

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
    # for datum in iter(ds_train):
    #     h, r, t = transe_model(datum)
    #     print(h)
    #     print(tf.norm(h))
    #     h /= tf.norm(h)
    #     print(h)
    #     print(tf.norm(h))
    #     break


if __name__ == "__main__":
    main()
