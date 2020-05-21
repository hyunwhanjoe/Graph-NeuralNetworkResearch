"""
TransE testing
https://github.com/muhaochen/MTransE/blob/master/run/en_de/test_MMTransE_lan_mapping_15k_fk.py
https://github.com/muhaochen/MTransE/blob/master/src/TransE/TransE.py
"""
import sys
import numpy as np
# sys.path.insert(0,'../../src/TransE')

from TransE import TransE

fmodel = "transe_test.bin"
model = TransE()
model.load(fmodel)


def seem_hit(x, y):
    for i in y:
        if x.find(i) > -1 or i.find(x) > -1:
            return True
    return False


fmap = "../../data/WK3l-15k/en_de/P_en_v6-test.csv"
topK = 10
past_num = 0
score = []

for line in open(fmap):
    line = line.rstrip('\n').split('@@@')
    if len(line) != 3:
        continue
    h = model.entity_vec(line[0])
    r = model.relation_vec(line[1])
    cand = model.kNN_entity(h + r)  # t - h
    cand = [x[0] for x in cand]
    print line[0], line[1], line[2]
    print cand

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
        print score
    else:
        score[0] = (score[0] * past_num + tmp_score) / (past_num + 1.0)
        print score
    past_num += 1

# sys.path.remove('')
# print sys.path
