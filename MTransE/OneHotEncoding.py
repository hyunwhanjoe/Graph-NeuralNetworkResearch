import numpy as np
from numpy import linalg as LA

index = {}
i = 0
path = "C:/Users/hyunwhanjoe/Downloads/MTransE data/CN3l/en_de/"
with open(path + "test.csv", "r") as file:
    for line in file:
        line = line.rstrip()
        triple = line.split("@@@")
        h = triple[0]
        r = triple[1]
        t = triple[2]

        if not index.get(h):
            index[h] = i
            i += 1

        if not index.get(r):
            index[r] = i
            i += 1

        if not index.get(t):
            index[t] = i
            i += 1
    file.seek(0)

    np.random.seed(1)
    embedding = np.random.random((len(index), len(index)))
    for line in file:
        line = line.rstrip()
        triple = line.split("@@@")

        h_onehot = np.zeros(len(index))
        h_onehot[index.get(triple[0])] = 1

        r_onehot = np.zeros(len(index))
        r_onehot[index.get(triple[1])] = 1

        t_onehot = np.zeros(len(index))
        t_onehot[index.get(triple[2])] = 1

        h = embedding.dot(h_onehot)
        r = embedding.dot(r_onehot)
        t = embedding.dot(t_onehot)
        norm = LA.norm(h + r - t)