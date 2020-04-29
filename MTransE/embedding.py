from tensorflow.keras import layers
import tensorflow as tf


def index(int_index, item, number):
    if not int_index.get(item):
        int_index[item] = number
        number += 1
        return number
    else:
        return number


indexDict = {}
i = 0
path = "data/CN3l/en_de/"

with open(path + "test.csv", "r") as file:
    for line in file:
        line = line.rstrip()
        triple = line.split("@@@")
        i = index(indexDict, triple[0], i)
        i = index(indexDict, triple[1], i)
        i = index(indexDict, triple[2], i)

    file.seek(0)
    vocabSize = len(indexDict)
    embedding_layer = layers.Embedding(vocabSize, 50)
    for line in file:
        line = line.rstrip()
        triple = line.split("@@@")
        h = embedding_layer(indexDict[triple[0]])
        r = embedding_layer(indexDict[triple[1]])
        t = embedding_layer(indexDict[triple[2]])
        print(tf.norm(h + r - t))
