import numpy as np
import tensorflow as tf

"""This function takes as input a triple csv file
   and returns a 3-tuple of numpy arrays representing
   the heads, relations and tails. The triple of arrays 
   will then be passed to a tensorflow from_tensor_slices
   method to make a dataset. There are two options for the
   method. 'one-hot' and 'integer' one-hot will return a
   tuple of numpy array one-hot vectors while integer will
   return a numpy array of integer encoded arrays"""


# one_hot_encode(path, delimiter = "@@@", encoding = "utf8"):
def encode(path, option):
    def index(index_dict, item, number):
        if not (item in index_dict):
            index_dict[item] = number
            number += 1
        return number

    def one_hot(line_num, entity_index, relation_index, file):
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
        return heads, relations, tails

    def integer_encode(line_num, entity_index, relation_index, file):
        triples = np.zeros((line_num, 3))
        current_line = 0
        file.seek(0)
        for line in file:
            line = line.rstrip()
            triple = line.split("@@@")

            integer = entity_index.get(triple[0])
            triples[current_line, 0] = integer
            integer = relation_index.get(triple[1])
            triples[current_line, 1] = integer
            integer = entity_index.get(triple[2])
            triples[current_line, 2] = integer

            current_line += 1
        return triples

    with open(path, "r", encoding="utf8") as file:
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

        if option == "one-hot":
            return one_hot(line_num, entity_index, relation_index, file)
        elif option == "integer":
            return integer_encode(line_num, entity_index, relation_index, file)
        else:
            print("incorrect option: one-hot, integer")
            return None


def main():
    # P_en_v6_training.csv
    test = "data/WK3l-15k/en_de/test.csv"
    # numpy = encode(test, "one-hot")
    numpy = encode(test, "integer")
    # print(numpy)
    dataset = tf.data.Dataset.from_tensor_slices(numpy)
    print(dataset)
    for data in dataset:
        print(data, data[0])


if __name__ == "__main__":
    main()
