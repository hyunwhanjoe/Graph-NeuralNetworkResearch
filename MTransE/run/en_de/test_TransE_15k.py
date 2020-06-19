import time
import tensorflow as tf
from MTransE.TransEHelper import TransEHelper
from MTransE.TransE_Embedding import TransEmbedding


def main():
    training_path = "../../data/WK3l-15k/en_de/P_en_v6_training.csv"
    test_path = "../../data/WK3l-15k/en_de/sample_test.csv"
    # test_path = "../../data/WK3l-15k/en_de/P_en_v6_test.csv"
    helper = TransEHelper()
    helper.generate_vocab(training_path)
    numpy = helper.encode_vocab(test_path)
    ds_test = tf.data.Dataset.from_tensor_slices(numpy).batch(32768)
    ds_test = ds_test.prefetch(tf.data.experimental.AUTOTUNE)

    model = TransEmbedding(helper.get_entity_count(), 75)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
    manager = tf.train.CheckpointManager(
        checkpoint, directory="../../models/en_de", max_to_keep=5)
    checkpoint.restore(manager.latest_checkpoint)

    for inputs in iter(ds_test):
        inputs = (inputs[:, 0], inputs[:, 1], inputs[:, 2])
        print(inputs)
        # print(helper.decode_relation(inputs[1][0]))
        # print(helper.decode_entity(inputs[2][0]))
        h, r, t = model(inputs)
        distance = tf.norm(h + r - t, axis=1)
        print(distance)


if __name__ == "__main__":
    t0 = time.time()
    main()
    print("Time used ", time.time() - t0)
