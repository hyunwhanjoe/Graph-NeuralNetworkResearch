import numpy as np


class TransEHelper(object):
    def __init__(self):
        self.entity_id = None
        self.id_entity = None
        self.relation_id = None
        self.id_relation = None
        self.entity_count = None
        self.missed_count = None

    def get_entity_count(self):
        return self.entity_count

    def get_missed_count(self):
        return self.missed_count

    def decode_entity(self, number):
        return self.id_entity[number]

    def decode_relation(self, number):
        return self.id_relation[number]

    # scan triple file once and encode vocabulary to integer values
    def generate_vocab(self, file_dir, delimiter='@@@', line_end='\n'):
        self.entity_id = {}
        self.id_entity = {}
        self.relation_id = {}
        self.id_relation = {}
        self.entity_count = 0
        relation_count = 0

        with open(file_dir, "r", encoding="utf8") as file:
            for line in file:
                line = line.rstrip(line_end).split(delimiter)
                if len(line) != 3:
                    print(line, " - triple does not have 3 elements")
                    continue

                if self.entity_id.get(line[0]) is None:
                    self.entity_id[line[0]] = self.entity_count
                    self.id_entity[self.entity_count] = line[0]
                    self.entity_count += 1

                if self.relation_id.get(line[1]) is None:
                    self.relation_id[line[1]] = relation_count
                    self.id_relation[relation_count] = line[1]
                    relation_count += 1

                if self.entity_id.get(line[2]) is None:
                    self.entity_id[line[2]] = self.entity_count
                    self.id_entity[self.entity_count] = line[2]
                    self.entity_count += 1

        print("entity count", self.entity_count)
        print("relation count", relation_count)

    # generate a numpy array representation of the triple file [[h,r,t], [h,r,t]]
    def encode_vocab(self, file_dir, delimiter='@@@', line_end='\n'):
        line_num = 0
        current_line = 0
        self.missed_count = 0
        with open(file_dir, "r", encoding="utf8") as file:
            # get line count
            for line in file:
                line_num += 1
            print("line count", line_num)
            file.seek(0)

            triples = np.zeros((line_num, 3))
            for line in file:
                triple = line.rstrip(line_end).split(delimiter)

                h, r, t = self.entity_id.get(triple[0]), self.relation_id.get(triple[1]), self.entity_id.get(triple[2])
                if h is None or r is None or t is None:
                    self.missed_count += 1
                else:
                    triples[current_line, 0] = h
                    triples[current_line, 1] = r
                    triples[current_line, 2] = t

                current_line += 1

            print("missed triples:", self.missed_count)
            return triples


def main():
    helper = TransEHelper()
    path = "data/WK3l-15k/en_de/test.csv"
    helper.generate_vocab(path)
    print(helper.entity_id)
    print(helper.relation_id)
    print(helper.encode_vocab(path))


if __name__ == "__main__":
    main()
