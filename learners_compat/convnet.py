import tensorflow as tf
# from tensorflow.contrib.layers.python import layers as tf_layers
import numpy as np
from utils import count_params_in_scope
from data import tf_standardize

# 5-layer convnet: from cbfinn's maml implementation


class ConvNet():

    def __init__(self, args):

        self.args = args
        self.channels = 3
        self.dim_hidden = [16, 32, 32, 64, 64]
        # self.dim_hidden = [16, 16, 16, 32, 32]
        self.dim_output = 10
        self.img_size = 32
        self.norm = 'None'
        self.max_pool = True
        self.floattype = tf.float64 if self.args.bit64 else tf.float32
        self.inttype = tf.int64 if self.args.bit64 else tf.int32

    def construct_weights(self):
        weights = {}

        conv_initializer = tf.contrib.layers.xavier_initializer_conv2d(dtype=self.floattype)
        fc_initializer = tf.contrib.layers.xavier_initializer(dtype=self.floattype)
        k = 3

        for i in range(len(self.dim_hidden)):
            previous = self.channels if i == 0 else self.dim_hidden[i - 1]
            weights['conv' + str(i + 1)] = tf.get_variable('conv' + str(i + 1), [k, k, previous,
                                                                                 self.dim_hidden[i]], initializer=conv_initializer, dtype=self.floattype)
            weights['b' + str(i + 1)] = tf.Variable(tf.zeros([self.dim_hidden[i]],
                                                             dtype=self.floattype), name='b' + str(i + 1))

        # assumes max pooling
        weights['w6'] = tf.get_variable('w6', [self.dim_hidden[-1], self.dim_output],
                                        initializer=fc_initializer, dtype=self.floattype)
        weights['b6'] = tf.Variable(tf.zeros([self.dim_output], dtype=self.floattype), name='b6')

        return weights

    def forward(self, inp, weights, reuse=False, scope=''):
        # reuse is for the normalization parameters.

        datamean, datastd = np.array((0.4914, 0.4822, 0.4465)), np.array((0.2023, 0.1994, 0.2010))
        inp = tf_standardize(inp, datamean, datastd)
        hidden1 = self.conv_block(inp, weights['conv1'], weights['b1'], reuse, scope + '0')
        hidden2 = self.conv_block(hidden1, weights['conv2'], weights['b2'], reuse, scope + '1')
        hidden3 = self.conv_block(hidden2, weights['conv3'], weights['b3'], reuse, scope + '2')
        hidden4 = self.conv_block(hidden3, weights['conv4'], weights['b4'], reuse, scope + '3')
        hidden5 = self.conv_block(hidden4, weights['conv5'], weights['b5'], reuse, scope + '4')

        hidden5 = tf.reshape(hidden5, [-1, np.prod([int(dim) for dim in hidden5.get_shape()[1:]])])
        logits = tf.matmul(hidden5, weights['w6']) + weights['b6']

        # nparam = count_params_in_scope()
        return logits, hidden5

    # Network helpers

    def conv_block(self, inp, cweight, bweight, reuse, scope, activation=tf.nn.relu, max_pool_pad='VALID', residual=False):
        """Perform, conv, batch norm, nonlinearity, and max pool."""
        stride, no_stride = [1, 2, 2, 1], [1, 1, 1, 1]

        if self.max_pool:
            conv_output = tf.nn.conv2d(inp, cweight, no_stride, 'SAME') + bweight
        else:
            conv_output = tf.nn.conv2d(inp, cweight, stride, 'SAME') + bweight
        normed = self.normalize(conv_output, activation, reuse, scope)
        if self.max_pool:
            normed = tf.nn.max_pool(normed, stride, stride, max_pool_pad)
        return normed

    def normalize(self, inp, activation, reuse, scope):
        if self.norm == 'batch_norm':
            return tf_layers.batch_norm(inp, activation_fn=activation, reuse=reuse, scope=scope)
        elif self.norm == 'layer_norm':
            return tf_layers.layer_norm(inp, activation_fn=activation, reuse=reuse, scope=scope)
        elif self.norm == 'None':
            if activation is not None:
                return activation(inp)
            else:
                return inp
