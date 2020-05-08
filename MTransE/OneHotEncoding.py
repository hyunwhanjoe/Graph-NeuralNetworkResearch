import numpy as np
import tensorflow as tf

"""This function takes as input a triple csv file
   and returns a 3-tuple of numpy arrays representing
   the heads, relations and tails. The triple of arrays 
   will then be passed to a tensorflow from_tensor_slices
   method to make a dataset. Each element per array
   represents the one hot encoding of the corresponding
   part in the triple"""


def one_hot_encode(path):
    def index(index_dict, item, number):
        if not (item in index_dict):
            index_dict[item] = number
            number += 1
        return number

    with open(path, "r") as file:
        indexes = {}
        i = 0
        line_num = 0
        for line in file:
            line = line.rstrip()
            triple = line.split(",")
            i = index(indexes, triple[0], i)
            i = index(indexes, triple[1], i)
            i = index(indexes, triple[2], i)
            line_num += 1
        index_max = len(indexes)
        heads = np.zeros((line_num, 1, index_max))
        # relations = np.zeros((line_num, index_max, 1))
        # tails = np.zeros((line_num, index_max, 1))
        current_line = 0
        file.seek(0)
        for line in file:
            line = line.rstrip()
            triple = line.split(",")

            one_hot = indexes.get(triple[0])
            heads[current_line, 0, one_hot] = 1

            # one_hot = indexes.get(triple[1])
            # relations[current_line, one_hot] = 1
            #
            # one_hot = indexes.get(triple[2])
            # tails[current_line, one_hot] = 1

            current_line += 1
        # print(heads)
        # print(relations)
        # print(tails)
        # print(entity_index)
        # print(relation_index)
        return heads


# test = "data/WK3l-15k/en_fr/test.csv"
# dataset = tf.data.Dataset.from_tensor_slices(one_hot_encode(test))
# for data in dataset.take(1):
#     print(data)


# file.seek(0)
# np.random.seed(1)
# embedding = np.random.random((len(index), len(index)))
# for line in file:
#     line = line.rstrip()
#     triple = line.split("@@@")
#
#     h_onehot = np.zeros(len(index))
#     h_onehot[index.get(triple[0])] = 1
#
#     r_onehot = np.zeros(len(index))
#     r_onehot[index.get(triple[1])] = 1
#
#     t_onehot = np.zeros(len(index))
#     t_onehot[index.get(triple[2])] = 1
#
#     h = embedding.dot(h_onehot)
#     r = embedding.dot(r_onehot)
#     t = embedding.dot(t_onehot)
#     norm = LA.norm(h + r - t)
