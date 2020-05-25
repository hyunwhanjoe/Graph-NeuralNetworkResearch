import sys
import time
import numpy as np

sys.path.insert(0, '../../src/TransE')

from TransE import TransE

start_time = time.time()
fmodel = "transe_test.bin"
model = TransE()
model.load(fmodel)


def modify_score(score, past_num, tmp_score):
    if len(score) == 0:
        score.append(tmp_score)
    else:
        score[0] = (score[0] * past_num + tmp_score) / (past_num + 1.0)


# fmap = "../../data/WK3l-15k/en_de/P_en_v6_sample_test.csv"
fmap = "../../data/WK3l-15k/en_de/P_en_v6_test.csv"
topK = 10
past_num = 0
score = []

for line in open(fmap):
    line = line.rstrip('\n').split('@@@')
    if len(line) != 3:
        continue
    h = model.entity_vec(line[0])
    r = model.relation_vec(line[1])
    if h is None or r is None:
        print("Head, relation is not in training set", line[0], line[1])
        tmp_score = np.zeros(topK)
        modify_score(score, past_num, tmp_score)
        past_num += 1
        continue

    cand = model.kNN_entity(h + r)  # t - h
    cand = [x[0] for x in cand]

    tmp_score = np.zeros(topK)
    hit = False
    last_i = 0
    tgt = line[2]
    if tgt is None:
        print("Tail is not in training set", line[2])
        tmp_score = np.zeros(topK)
        modify_score(score, past_num, tmp_score)
        past_num += 1
        continue

    for i in range(len(cand)):
        tmp_cand = cand[i]
        if (hit is False) and (tmp_cand == tgt):
            hit = True
        if hit:
            tmp_score[i] = 1.0

    modify_score(score, past_num, tmp_score)

    if past_num % 100 == 0:
        print(past_num, line[0], line[1], line[2])
        print(cand)
        print(score)
        print(time.time() - start_time)
    past_num += 1

print(score)
print(time.time() - start_time)