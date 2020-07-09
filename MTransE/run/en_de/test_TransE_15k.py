"""
Adopted from
https://github.com/muhaochen/MTransE/blob/master/run/en_de/test_MMTransE_lan_mapping_15k_fk.py
"""
import time
import tensorflow as tf
import numpy as np
from MTransE.TransEHelper import TransEHelper
from MTransE.TransE_Embedding import TransEmbedding


def test(model, helper, ds_test):
    def remove_head_from_cand(top10cand, all_cand, h_index, k):
        for j, e_index in enumerate(top10cand):
            if h_index == e_index:
                t0 = all_cand[:j]
                t1 = all_cand[j+1:k+1]
                return tf.concat([t0, t1], 0)
        return top10cand

    def modify_score(scores, past_nums, tmp_scores):
        if len(scores) == 0:
            scores.append(tmp_scores)
        else:
            scores[0] = (scores[0] * past_nums + tmp_scores) / (past_nums + 1.0)

    time0 = time.time()

    entity_count = helper.get_entity_count()
    indexes = tf.convert_to_tensor(np.arange(entity_count))
    entity_embeddings, p, o = model((indexes, indexes, indexes))
    entity_embeddings = tf.math.l2_normalize(entity_embeddings, axis=1)

    top_k = 10
    past_num = 0
    score = []

    for inputs in ds_test:
        inputs = (inputs[:, 0], inputs[:, 1], inputs[:, 2])

        h, r, t = model(inputs)
        h = tf.math.l2_normalize(h)
        h_plus_r = h + r

        # tf.reduce_sum((pos_h_e + pos_r_e - pos_t_e) ** 2, 1, keep_dims = True)
        # kNN
        distance = tf.reduce_sum(tf.pow((h_plus_r - entity_embeddings), 2.0), -1)
        sorted_distances = tf.argsort(distance, axis=-1, direction='ASCENDING')
        cand = sorted_distances[:top_k]
        head_index = inputs[0].numpy()[0]
        cand = remove_head_from_cand(cand, sorted_distances, head_index, top_k)
        cand = cand.numpy()

        tmp_score = np.zeros(top_k)
        hit = False
        tgt = inputs[2].numpy()[0]

        for i in range(len(cand)):
            if (hit is False) and (cand[i] == tgt):
                hit = True
            if hit:
                tmp_score[i] = 1.0

        modify_score(score, past_num, tmp_score)

        if past_num % 100 == 0:
            print(past_num, helper.decode_entity(head_index),
                  helper.decode_relation(inputs[1].numpy()[0]),
                  helper.decode_entity(inputs[2].numpy()[0]))
            for entity_index in cand:
                print(helper.decode_entity(entity_index), entity_index)
            print("score:", score)
            print("Time used ", time.time() - time0, "\n")

        past_num += 1

# account for missed triples
    print("Missed triple:", helper.get_missed_count())
    print("Pre-missed score:", score)
    tmp_score = np.zeros(top_k)
    for i in range(helper.get_missed_count()):
        modify_score(score, past_num, tmp_score)
        past_num += 1
    print("score:", score)


def main():
    training_path = "../../data/WK3l-15k/en_de/P_en_v6_training.csv"
    # test_path = "../../data/WK3l-15k/en_de/sample_test.csv"/
    test_path = "../../data/WK3l-15k/en_de/P_en_v6_test.csv"
    helper = TransEHelper()
    helper.generate_vocab(training_path)
    numpy = helper.encode_vocab(test_path)
    ds_test = tf.data.Dataset.from_tensor_slices(numpy).batch(1)
    ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

    model = TransEmbedding(helper.get_entity_count(), 75)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)
    # match optimizer and model
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
    manager = tf.train.CheckpointManager(
        checkpoint, directory="../../models/en_de", max_to_keep=5)
    checkpoint.restore(manager.latest_checkpoint)

    test(model, helper, ds_test)


if __name__ == "__main__":
    main()
