import time
import tensorflow as tf

from MTransE.Encoding import encode
from tensorflow.keras import Model
from tensorflow.keras import layers


# https://www.tensorflow.org/guide/keras/custom_layers_and_models
class TransELayer(layers.Layer):
    def __init__(self, units):
        super(TransELayer, self).__init__()
        self.w = None
        self.b = None
        self.units = units

    def build(self, input_shape):
        w_init = tf.random_normal_initializer(mean=0., stddev=1.)
        self.w = tf.Variable(initial_value=w_init(shape=(input_shape[-1], self.units),
                                                  dtype='float32'),
                             trainable=True)
        self.w = tf.math.l2_normalize(self.w, axis=1)

    def call(self, inputs):
        return tf.matmul(inputs, self.w)  # + self.b


class TransE(Model):
    def __init__(self):
        super(TransE, self).__init__()
        self.rembedding = None
        self.eembedding = None

    def build(self, input_shape):  # input_shape is needed
        self.rembedding = TransELayer(75)
        self.eembedding = TransELayer(75)

    def call(self, inputs, **kwargs):
        h, r, t = inputs
        return self.eembedding(h), self.rembedding(r), self.eembedding(t)


def main():
    t0 = time.time()
    print(tf.__version__)
    path = "data/WK3l-15k/en_de/test.csv"
    ds_train = tf.data.Dataset.from_tensor_slices(encode(path, "one-hot"))  # mini-batch?
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

    # for epoch in range(1, 20):
    #     epoch_loss_total = tf.keras.metrics.Sum()
    #     for datum in iter(ds_train):
    #         train(datum, epoch_loss_total)
    #     print("Epoch %d loss %.3f" % (epoch, epoch_loss_total.result()))
    # print("Time used ", time.time() - t0)
    for datum in iter(ds_train):
        h, r, t = transe_model(datum)
        print(tf.norm(h))
        print(tf.norm(r))
        print(tf.norm(t))
        break


if __name__ == "__main__":
    main()
