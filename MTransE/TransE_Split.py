"""
Split into training and test set
"""
import random
path = "data/WK3l-15k/en_de/P_en_v6.csv"
triples = []
with open(path, "r", encoding="utf8") as file:
    for line in file:
        line = line.rstrip()
        triples.append(line)
    random.shuffle(triples)

test_path = "data/WK3l-15k/en_de/P_en_v6_test.csv"
training_path = "data/WK3l-15k/en_de/P_en_v6_training.csv"
with open(test_path, "w", encoding="utf8") as test, \
        open(training_path, "w", encoding="utf8") as training:
    test_percent = 0.1
    i = 0
    while i < (len(triples) * 0.1):
        test.write(triples[i]+"\n")
        i += 1

    print(i)
    while i < len(triples):
        training.write(triples[i]+"\n")
        i += 1
