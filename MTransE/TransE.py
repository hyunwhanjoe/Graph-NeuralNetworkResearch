import tensorflow as tf


class KGEmbeddingNeuralNetwork(tf.keras.layers.Layer):
    def __init__(self):
        super(KGEmbeddingNeuralNetwork, self).__init__()
        self.embedding = None

    def build(self):
        self.embedding = tf.keras.layers.Dense(75, activation="relu")

    def call(self, inputs, **kwargs):
        return self.embedding(inputs)


def main():
    print(tf.__version__)

    entity_model = KGEmbeddingNeuralNetwork()
    relation_model = KGEmbeddingNeuralNetwork()
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
