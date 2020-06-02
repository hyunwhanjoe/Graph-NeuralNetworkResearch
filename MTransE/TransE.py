import tensorflow as tf

from MTransE.OneHotEncoding import one_hot_encode
from tensorflow.keras import Model


class TransE(Model):
    def __init__(self):
        super(TransE, self).__init__()
        self.rembedding = None
        self.eembedding = None

    def build(self, input_shape):  # input_shape is needed
        self.rembedding = tf.keras.layers.Dense(128, activation="relu")
        self.eembedding = tf.keras.layers.Dense(128, activation="relu")

    def call(self, inputs, **kwargs):
        s, p, o = inputs
        return self.eembedding(s), self.rembedding(p), self.eembedding(o)


def main():
    print(tf.__version__)
    path = "data/WK3l-15k/en_fr/test.csv"
    ds_train = tf.data.Dataset.from_tensor_slices(one_hot_encode(path))

    transe_model = TransE()
    optimizer = tf.keras.optimizers.Adam()

    @tf.function
    def train(inputs, loss_state):
        with tf.GradientTape() as tape:
            h, r, t = transe_model(inputs)
            loss = tf.norm(h + r - t)
            grads = tape.gradient(loss, transe_model.trainable_variables)
            optimizer.apply_gradients(zip(grads, transe_model.trainable_variables))
        loss_state.update_state(loss)

    for epoch in range(1, 20):
        epoch_loss_avg = tf.keras.metrics.Sum()
        for datum in iter(ds_train):
            train(datum, epoch_loss_avg)
        print("Epoch %d loss %.3f" % (epoch, epoch_loss_avg.result()))


if __name__ == "__main__":
    main()
