"""
Split into training and test set
Double check for off by one errors
"""
import random
path = "../../data/WK3l-15k/en_de/P_en_v6.csv"
triples = []
with open(path, "r") as file:
    for line in file:
        line = line.rstrip()
        triples.append(line)
    random.shuffle(triples)

test_path = "../../data/WK3l-15k/en_de/P_en_v6_test.csv"
test = open(test_path, "w")
training_path = "../../data/WK3l-15k/en_de/P_en_v6_training.csv"
training = open(training_path, "w")

test_percent = 0.1
i = 0
while i < len(triples) * 0.1:
    test.write(triples[i]+"\n")
    i += 1
test.close()

print(i)
while i < len(triples):
    training.write(triples[i]+"\n")
    i += 1
training.close()
