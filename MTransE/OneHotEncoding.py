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
        entity_index = {}
        relation_index = {}
        entity_i = 0
        relation_i = 0
        line_num = 0
        for line in file:
            line = line.rstrip()
            triple = line.split(",")
            entity_i = index(entity_index, triple[0], entity_i)
            relation_i = index(relation_index, triple[1], relation_i)
            entity_i = index(entity_index, triple[2], entity_i)
            line_num += 1

        heads = np.zeros((line_num, len(entity_index)))
        relations = np.zeros((line_num, len(relation_index)))
        tails = np.zeros((line_num, len(entity_index)))
        current_line = 0
        file.seek(0)
        for line in file:
            line = line.rstrip()
            triple = line.split(",")

            one_hot = entity_index.get(triple[0])
            heads[current_line, one_hot] = 1

            one_hot = relation_index.get(triple[1])
            relations[current_line, one_hot] = 1

            one_hot = entity_index.get(triple[2])
            tails[current_line, one_hot] = 1

            current_line += 1
        # print(heads)
        # print(relations)
        # print(tails)
        # print(entity_index)
        # print(relation_index)
        return heads, relations, tails


test = "data/WK3l-15k/en_fr/test.csv"
ds_train = tf.data.Dataset.from_tensor_slices(one_hot_encode(test))
for elem in ds_train:
    print(elem)
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
