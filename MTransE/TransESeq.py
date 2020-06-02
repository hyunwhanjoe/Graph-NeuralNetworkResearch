import tensorflow as tf
from MTransE.OneHotEncoding import one_hot_encode

transe_model = tf.keras.Sequential([
    tf.keras.layers.Dense(25, activation="relu")
])
path = "data/WK3l-15k/en_fr/test.csv"
dataset = tf.data.Dataset.from_tensor_slices(one_hot_encode(path))
for data in dataset.take(1):
    print(transe_model(data[0]))
    print(transe_model(data[1]))
    print(transe_model(data[2]))
