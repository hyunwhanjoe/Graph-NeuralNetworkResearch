import sys
import time
import numpy as np

sys.path.insert(0, '../../src/TransE')

from TransE import TransE

start_time = time.time()
fmodel = "transe1.bin"
model = TransE()
model.load(fmodel)


def modify_score(score, past_num, tmp_score):
    if len(score) == 0:
        score.append(tmp_score)
    else:
        score[0] = (score[0] * past_num + tmp_score) / (past_num + 1.0)


fmap = "../../data/STRINGS/acting_test.csv"
topK = 10
past_num = 0
missing = 0
score = []

for line in open(fmap):
    line = line.rstrip('\n').split('@@@')
    if len(line) != 3:
        continue
    h = model.entity_vec(line[0])
    r = model.relation_vec(line[1])
    t = model.entity_vec(line[2])
    if h is None or r is None or t is None:
        missing += 1
        print("Entity or relation is not in training set", line)
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

    for i in range(len(cand)):
        tmp_cand = cand[i]
        if (hit is False) and (tmp_cand == tgt):
            hit = True
        if hit:
            tmp_score[i] = 1.0

    modify_score(score, past_num, tmp_score)

    if past_num % 100 == 0:
        print(past_num, line)
        print(cand)
        print(score)
        print(time.time() - start_time)
    past_num += 1

print("score", score)
print("missing", missing)
print(time.time() - start_time)