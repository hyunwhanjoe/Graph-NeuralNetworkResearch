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
        entity_count = 0
        relation_count = 0
        line_num = 0
        for line in file:
            line = line.rstrip()
            triple = line.split("@@@")
            entity_count = index(entity_index, triple[0], entity_count)
            relation_count = index(relation_index, triple[1], relation_count)
            entity_count = index(entity_index, triple[2], entity_count)
            line_num += 1

        index_max = len(entity_index)  # match entity and relation dimensions
        heads = np.zeros((line_num, 1, index_max))
        relations = np.zeros((line_num, 1, index_max))
        tails = np.zeros((line_num, 1, index_max))
        current_line = 0
        file.seek(0)

        for line in file:
            line = line.rstrip()
            triple = line.split("@@@")

            one_hot = entity_index.get(triple[0])
            heads[current_line, 0, one_hot] = 1

            one_hot = relation_index.get(triple[1])
            relations[current_line, 0, one_hot] = 1

            one_hot = entity_index.get(triple[2])
            tails[current_line, 0, one_hot] = 1

            current_line += 1
        # print(heads)
        # print(relations)
        # print(tails)
        # print(entity_index)
        # print(relation_index)
        return heads, relations, tails


test = "data/WK3l-15k/en_de/test.csv"
dataset = tf.data.Dataset.from_tensor_slices(one_hot_encode(test))
for data in dataset:
    print(data)
