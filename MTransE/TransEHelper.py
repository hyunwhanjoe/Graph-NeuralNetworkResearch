import numpy as np


class TransEHelper(object):
    def __init__(self):
        self.entity_id = None
        self.relation_id = None
        self.entity_count = None

    def get_entity_count(self):
        return self.entity_count

    # scan triple file once and encode vocabulary to integer values
    def generate_vocab(self, file_dir, delimiter='@@@', line_end='\n'):
        self.entity_id = {}
        self.relation_id = {}
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
                    self.entity_count += 1
                if self.relation_id.get(line[1]) is None:
                    self.relation_id[line[1]] = relation_count
                    relation_count += 1
                if self.entity_id.get(line[2]) is None:
                    self.entity_id[line[2]] = self.entity_count
                    self.entity_count += 1

        print("entity count", self.entity_count)
        print("relation count", relation_count)

    # generate a numpy array representation of the triple file [[h,r,t], [h,r,t]]
    def encode_vocab(self, file_dir, delimiter='@@@', line_end='\n'):
        line_num = 0
        current_line = 0
        with open(file_dir, "r", encoding="utf8") as file:
            # get line count
            for line in file:
                line_num += 1
            print("line count", line_num)
            file.seek(0)

            triples = np.zeros((line_num, 3))
            for line in file:
                triple = line.rstrip(line_end).split(delimiter)

                encoding = self.entity_id.get(triple[0])
                triples[current_line, 0] = encoding
                encoding = self.relation_id.get(triple[1])
                triples[current_line, 1] = encoding
                encoding = self.entity_id.get(triple[2])
                triples[current_line, 2] = encoding

                current_line += 1

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
