# -*- coding:utf-8 -*-
"""cnn_mnist.py: the simplest example for the custom training loop"""
import tensorflow as tf
import tensorflow_datasets as tfds

__author__ = "Kyungmin Lee"
__email__ = "sephiroce@snu.ac.kr"


class ConvolutionalNeuralNetwork(tf.keras.layers.Layer):
    def __init__(self):
        super(ConvolutionalNeuralNetwork, self).__init__()
        # self.cnn = None
        self.flat = None
        self.dense1 = None
        self.dense2 = None

    def build(self, input_shape):  # input_shape not used
        # self.cnn = tf.keras.layers.Conv2D(input_shape=input_shape,
        #                                  filters=12, kernel_size=[3, 3],
        #                                  padding="same",
        #                                  activation="relu")
        self.flat = tf.keras.layers.Flatten()  # flattens input
        self.dense1 = tf.keras.layers.Dense(1024, activation="relu")
        self.dense2 = tf.keras.layers.Dense(10)  # since Mnist has 10 classes

    def call(self, inputs, **kwargs):
        # [batch, 28, 28, 1]
        inputs = self.flat(tf.squeeze(inputs, -1))

        return self.dense2(self.dense1(inputs))


def main():
    # pylint: disable=too-many-locals
    print(tf.__version__)
    mnist_builder = tfds.builder("mnist")
    mnist_builder.download_and_prepare()
    ds_train = mnist_builder.as_dataset(split="train")
    ds_test = mnist_builder.as_dataset(split="test")

    ds_train = ds_train.repeat(1).shuffle(1024).batch(32)
    ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

    ds_test = ds_test.repeat(1).batch(32)
    ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

    cnn_model = ConvolutionalNeuralNetwork()
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    # cnn_model.build(input_shape=[-1, 28, 28, 1])

    loss_object = \
        tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

    @tf.function
    def train(inputs, loss_state, acc_state):
        input_image = tf.cast(inputs["image"], tf.float32)
        input_label = tf.cast(inputs["label"], tf.float32)

        with tf.GradientTape() as tape:
            y_pred = cnn_model(input_image)
            loss = loss_object(input_label, y_pred)
            grads = tape.gradient(loss, cnn_model.trainable_variables)
            optimizer.apply_gradients(zip(grads, cnn_model.trainable_variables))
        loss_state.update_state(loss)
        acc_state.update_state(input_label, y_pred)

    for epoch in range(1, 5):
        epoch_loss_avg = tf.keras.metrics.Mean()
        epoch_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()
        for datum in iter(ds_train):
            train(datum, epoch_loss_avg, epoch_accuracy)
        print("Epoch %d loss %.3f, accuracy %.3f" % (epoch, epoch_loss_avg.result(),
                                                     epoch_accuracy.result()))

    loss_state = tf.keras.metrics.Mean()
    acc_state = tf.keras.metrics.SparseCategoricalAccuracy()
    for inputs in iter(ds_test):
        input_image = tf.cast(inputs["image"], tf.float32)
        input_label = tf.cast(inputs["label"], tf.float32)
        y_pred = cnn_model(input_image)
        loss = loss_object(input_label, y_pred)
        loss_state.update_state(loss)
        acc_state.update_state(input_label, y_pred)
    print("Test loss %.3f, accuracy %.3f" % (loss_state.result(),
                                             acc_state.result()))


if __name__ == "__main__":
    main()
