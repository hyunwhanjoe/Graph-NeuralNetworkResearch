import time
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow_core.python.keras.layers import Embedding

from MTransE.TransEHelper import TransEHelper
from MTransE.TransE_Layers import RelationLayer, EntityLayer


class TransEmbedding(tf.keras.Model):
    def __init__(self, entity_count, dimensions):
        super(TransEmbedding, self).__init__()
        self.rembedding = None
        self.eembedding = None
        self.entity_count = entity_count
        self.dimensions = dimensions

    def build(self, input_shape):
        self.rembedding = RelationLayer(self.entity_count, self.dimensions)
        self.eembedding = Embedding(self.entity_count, self.dimensions)

    def call(self, inputs, **kwargs):
        # h = tf.reshape(inputs[0], [-1])  # make 1-d
        # r = tf.reshape(inputs[1], [-1])
        # t = tf.reshape(inputs[2], [-1])
        h = inputs[0]
        r = inputs[1]
        t = inputs[2]
        return self.eembedding(h), self.rembedding(r), self.eembedding(t)


def main():
    # tf.keras.backend.set_floatx('float64')
    print(tf.__version__)
    path = "data/WK3l-15k/en_de/P_en_v6_training.csv"
    # path = "data/WK3l-15k/en_de/test.csv"
    helper = TransEHelper()
    helper.generate_vocab(path)
    numpy = helper.encode_vocab(path)
    # warnings for batch sizes higher than 16,384
    ds_train = tf.data.Dataset.from_tensor_slices(numpy).shuffle(1024).batch(8192)
    ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)
    model = TransEmbedding(helper.get_entity_count(), 75)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)

    @tf.function
    def train(inputs, loss_state):
        with tf.GradientTape() as tape:
            inputs = (inputs[:, 0], inputs[:, 1], inputs[:, 2])  # input as tuple
            h, r, t = model(inputs)
            # loss = tf.reduce_sum(tf.pow((h + r - t), 2.0), -1)
            h = tf.math.l2_normalize(h, axis=1)
            t = tf.math.l2_normalize(t, axis=1)
            loss = tf.norm(h + r - t, axis=1)
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            loss_state.update_state(loss)

    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
    manager = tf.train.CheckpointManager(
        checkpoint, directory="models/en_de", max_to_keep=5)
    checkpoint.restore(manager.latest_checkpoint)

    for epoch in range(1, 401):
        epoch_loss_total = tf.keras.metrics.Sum()
        for datum in iter(ds_train):
            train(datum, epoch_loss_total)
        print("%d\t%.3f" % (epoch, epoch_loss_total.result()))
        if epoch_loss_total.result() < 200:
            break

    manager.save()


if __name__ == "__main__":
    t0 = time.time()
    main()
    print("Time used ", time.time() - t0)
