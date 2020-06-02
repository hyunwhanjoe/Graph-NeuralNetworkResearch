import tensorflow as tf
import tensorflow_datasets as tfds

mnist_builder = tfds.builder("mnist")
mnist_builder.download_and_prepare()
ds_train = mnist_builder.as_dataset(split="train")
iterator = ds_train.make_one_shot_iterator()
next_element = iterator.get_next()
print(next_element)