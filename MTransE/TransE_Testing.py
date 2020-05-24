"""
TransE testing
"""
import sys
import time
import numpy as np

sys.path.insert(0, '../../src/TransE')
print(sys.path)

from TransE import TransE

start_time = time.time()
fmodel = "transe_test.bin"
model = TransE()
model.load(fmodel)


def seem_hit(x, y):
    for i in y:
        if x.find(i) > -1 or i.find(x) > -1:
            return True
    return False


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
        print("Is not in training set", line[0], line[1])
        continue
    cand = model.kNN_entity(h + r)  # t - h
    cand = [x[0] for x in cand]
    print(past_num, line[0], line[1], line[2])
    print(cand)

    tmp_score = np.zeros(topK)
    hit = False
    last_i = 0
    tgt = line[2]
    if tgt == None:
        continue

    for i in range(len(cand)):
        last_i = i
        tmp_cand = cand[i]
        if hit == False and (seem_hit(tmp_cand, tgt) == True):
            hit = True
        if hit == True:
            tmp_score[i] = 1.0

    while last_i < topK:  # dont know what the purpose for this is
        if hit:
            tmp_score[last_i] = 1.0
        last_i += 1

    if len(score) == 0:
        score.append(tmp_score)
    else:
        score[0] = (score[0] * past_num + tmp_score) / (past_num + 1.0)
    print(score)
    past_num += 1

# sys.path.remove('')
# print sys.path
print(time.time() - start_time)
