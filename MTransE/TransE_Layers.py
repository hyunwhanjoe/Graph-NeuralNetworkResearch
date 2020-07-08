# https://github.com/tensorflow/tensorflow/blob/v2.2.0/tensorflow/python/keras/layers/embeddings.py
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.python.eager import context
from tensorflow.python.framework import ops
from tensorflow.python.keras import backend as K
from tensorflow.python.keras import constraints
from tensorflow.python.keras import initializers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.util.tf_export import keras_export


class RelationLayer(Layer):
    def __init__(self,
                 input_dim,
                 output_dim,
                 embeddings_initializer='uniform',
                 embeddings_regularizer=None,
                 activity_regularizer=None,
                 embeddings_constraint=None,
                 mask_zero=False,
                 input_length=None,
                 **kwargs):
        if 'input_shape' not in kwargs:
            if input_length:
                kwargs['input_shape'] = (input_length,)
            else:
                kwargs['input_shape'] = (None,)
        dtype = kwargs.pop('dtype', K.floatx())
        # We set autocast to False, as we do not want to cast floating- point inputs
        # to self.dtype. In call(), we cast to int32, and casting to self.dtype
        # before casting to int32 might cause the int32 values to be different due
        # to a loss of precision.
        kwargs['autocast'] = False
        super(RelationLayer, self).__init__(dtype=dtype, **kwargs)

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.embeddings_initializer = initializers.get(embeddings_initializer)
        self.embeddings_regularizer = regularizers.get(embeddings_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)
        self.embeddings_constraint = constraints.get(embeddings_constraint)
        self.mask_zero = mask_zero
        self.supports_masking = mask_zero
        self.input_length = input_length
        self._supports_ragged_inputs = True

    @tf_utils.shape_type_conversion
    def build(self, input_shape):
        # Note: most sparse optimizers do not have GPU kernels defined. When
        # building graphs, the placement algorithm is able to place variables on CPU
        # since it knows all kernels using the variable only exist on CPU.
        # When eager execution is enabled, the placement decision has to be made
        # right now. Checking for the presence of GPUs to avoid complicating the
        # TPU codepaths which can handle sparse optimizers.
        if context.executing_eagerly() and context.context().num_gpus():
            with ops.device('cpu:0'):
                self.embeddings = self.add_weight(
                    shape=(self.input_dim, self.output_dim),
                    initializer=self.embeddings_initializer,
                    name='embeddings',
                    regularizer=self.embeddings_regularizer,
                    constraint=self.embeddings_constraint)
        else:
            self.embeddings = self.add_weight(
                shape=(self.input_dim, self.output_dim),
                initializer=self.embeddings_initializer,
                name='embeddings',
                regularizer=self.embeddings_regularizer,
                constraint=self.embeddings_constraint)
        self.built = True
        # print(self.embeddings)
        # print("before", tf.norm(self.embeddings[1]))

        self.embeddings = tf.math.l2_normalize(self.embeddings, axis=1)
        # print(self.embeddings)
        # print("after", tf.norm(self.embeddings[1]))


    def compute_mask(self, inputs, mask=None):
        if not self.mask_zero:
            return None

        return math_ops.not_equal(inputs, 0)

    @tf_utils.shape_type_conversion
    def compute_output_shape(self, input_shape):
        if self.input_length is None:
            return input_shape + (self.output_dim,)
        else:
            # input_length can be tuple if input is 3D or higher
            if isinstance(self.input_length, (list, tuple)):
                in_lens = list(self.input_length)
            else:
                in_lens = [self.input_length]
            if len(in_lens) != len(input_shape) - 1:
                raise ValueError('"input_length" is %s, '
                                 'but received input has shape %s' % (str(
                    self.input_length), str(input_shape)))
            else:
                for i, (s1, s2) in enumerate(zip(in_lens, input_shape[1:])):
                    if s1 is not None and s2 is not None and s1 != s2:
                        raise ValueError('"input_length" is %s, '
                                         'but received input has shape %s' % (str(
                            self.input_length), str(input_shape)))
                    elif s1 is None:
                        in_lens[i] = s2
            return (input_shape[0],) + tuple(in_lens) + (self.output_dim,)

    def call(self, inputs):
        dtype = K.dtype(inputs)
        if dtype != 'int32' and dtype != 'int64':
            inputs = math_ops.cast(inputs, 'int32')
        out = embedding_ops.embedding_lookup(self.embeddings, inputs)
        return out

    def get_config(self):
        config = {
            'input_dim': self.input_dim,
            'output_dim': self.output_dim,
            'embeddings_initializer':
                initializers.serialize(self.embeddings_initializer),
            'embeddings_regularizer':
                regularizers.serialize(self.embeddings_regularizer),
            'activity_regularizer':
                regularizers.serialize(self.activity_regularizer),
            'embeddings_constraint':
                constraints.serialize(self.embeddings_constraint),
            'mask_zero': self.mask_zero,
            'input_length': self.input_length
        }
        base_config = super(RelationLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def main():
    r_layer = RelationLayer(2, 75)
    print(r_layer(0))


if __name__ == "__main__":
    main()
