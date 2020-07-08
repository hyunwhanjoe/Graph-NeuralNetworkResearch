import unittest
import tensorflow as tf
from MTransE.TransEHelper import TransEHelper
from MTransE.TransE_Embedding import TransEmbedding


class MyTestCase(unittest.TestCase):
    def test_normalization(self):
        def test(embedding):
            norm = tf.norm(embedding).numpy()
            print(norm)
            if 0.9 < norm < 1.1:
                return True
            return False

        path = "../data/WK3l-15k/en_de/P_en_v6_training.csv"
        helper = TransEHelper()
        helper.generate_vocab(path)
        numpy = helper.encode_vocab(path)
        ds_train = tf.data.Dataset.from_tensor_slices(numpy).shuffle(1024).batch(8192)
        ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)
        model = TransEmbedding(helper.get_entity_count(), 75)
        result = False
        for inputs in iter(ds_train):
            inputs = (inputs[:, 0], inputs[:, 1], inputs[:, 2])
            h, r, t = model(inputs)
            # result = test(h[0])
            result = test(r[0])
            break

        self.assertEqual(True, result)


if __name__ == '__main__':
    unittest.main()
