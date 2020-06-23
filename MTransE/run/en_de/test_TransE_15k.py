import time
import tensorflow as tf
import numpy as np
from MTransE.TransEHelper import TransEHelper
from MTransE.TransE_Embedding import TransEmbedding


def test(model, helper, ds_test):
    entity_count = helper.get_entity_count()
    indexes = tf.convert_to_tensor(np.arange(entity_count))
    all_entity_embeddings, p, o = model((indexes, indexes, indexes))
    topK = 10
    past_num = 0
    score = []

    for inputs in iter(ds_test):
        inputs = (inputs[:, 0], inputs[:, 1], inputs[:, 2])

        h_index = int(inputs[0].numpy()[0])
        t0 = all_entity_embeddings[:h_index]
        t1 = all_entity_embeddings[h_index + 1:]
        print("all", all_entity_embeddings)
        print("t0", t0)
        print("t1", t1)
        entity_embeddings = tf.concat([t0, t1], 0)
        print(entity_embeddings)
        h, r, t = model(inputs)
        h_plus_r = h + r

        distance = tf.reduce_sum(tf.pow((h_plus_r - entity_embeddings), 2.0), -1)
        cand = tf.argsort(distance, axis=-1, direction='ASCENDING')[:10]
        print(helper.decode_entity(inputs[0].numpy()[0]))
        print(helper.decode_relation(inputs[1].numpy()[0]))
        print(helper.decode_entity(inputs[2].numpy()[0]), "\n")
        # bug
        for index in cand:
            entity_index = index.numpy()
            if entity_index >= h_index:
                print(helper.decode_entity(entity_index+1))
            else:
                print(helper.decode_entity(entity_index))


def main():
    training_path = "../../data/WK3l-15k/en_de/P_en_v6_training.csv"
    test_path = "../../data/WK3l-15k/en_de/sample_test.csv"
    # test_path = "../../data/WK3l-15k/en_de/P_en_v6_test.csv"
    helper = TransEHelper()
    helper.generate_vocab(training_path)
    numpy = helper.encode_vocab(test_path)
    ds_test = tf.data.Dataset.from_tensor_slices(numpy).batch(1)
    ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

    model = TransEmbedding(helper.get_entity_count(), 75)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
    manager = tf.train.CheckpointManager(
        checkpoint, directory="../../models/en_de/400", max_to_keep=5)
    checkpoint.restore(manager.latest_checkpoint)

    test(model, helper, ds_test)


if __name__ == "__main__":
    t0 = time.time()
    main()
    print("Time used ", time.time() - t0)
