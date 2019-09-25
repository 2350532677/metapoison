# Copyright 2018 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
# Adapted TF Resnets from
# JOG
#
#
"""ResNet56 model for Keras adapted from tf.keras.applications.ResNet50.

# Reference:
- [Deep Residual Learning for Image Recognition](
    https://arxiv.org/abs/1512.03385)
Adapted from code contributed by BigMoyan.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # silence tensorflow
import tensorflow as tf
import numpy as np
from tensorflow.python.keras import backend
from tensorflow.python.keras import initializers
from tensorflow.python.keras import layers
from tensorflow.python.keras import regularizers

from data import tf_standardize

from .modules import Conv2D, BatchNormalization, Dense

from collections import defaultdict


BATCH_NORM_DECAY = 0.997
BATCH_NORM_EPSILON = 1e-5
L2_WEIGHT_DECAY = 2e-4


class ResNet():
    """ResNet Class.

    This class wraps the TF keras ResNet implementation to trace the model without external parameters. This produces
    a list of weights that can then be used to directly call the ResNet with a given list of weights.
    This approach should work for any net?

    TF Keras ResNet adapted from https://github.com/tensorflow/models/blob/master/official/vision/image_classification/resnet_cifar_main.py
    """

    def __init__(self, args, num_blocks=3, classes=10):
        self.droprate = args.droprate
        self.num_blocks = num_blocks
        self.classes = classes
        self.args = args
        if self.args.bit64:
            raise NotImplementedError()

    def infer_weight_dict(self):
        """Infer weights by tracing a default ResNet and then collecting its parameters."""
        params = defaultdict(lambda: None)
        model = self.resnet(self.num_blocks, params=params,
                            img_input=None, classes=self.classes, training=True)
        current_scope = tf.get_default_graph().get_name_scope()
        scope_len = len(current_scope) + 1
        params = {param.name[scope_len:] : param for param in model.trainable_weights}
        # for name, param in dict.items():
        #     print(name, param)
        return params

    def construct_weights(self):
        """Construct weights from traced parameters."""
        weights = self.infer_weight_dict()
        return weights

    def forward(self, inputs, weights):  # reuse=False, scope='' ?
        """Call Resnet functionally and return output tensor for given input and given weights."""
        datamean, datastd = np.array((0.4914, 0.4822, 0.4465)), np.array((0.2023, 0.1994, 0.2010))
        inputs = tf_standardize(inputs, datamean, datastd)
        output = self.resnet(self.num_blocks, weights, img_input=inputs, classes=self.classes, training=True)
        return output

    def identity_building_block(self, input_tensor, params,
                                kernel_size,
                                filters,
                                stage,
                                block,
                                training=None):
        """The identity block is the block that has no conv layer at shortcut.

        Arguments:
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: current block label, used for generating layer names
        training: Only used if training keras model with Estimator.  In other
          scenarios it is handled automatically.

        Returns:
        Output tensor for the block.

        """
        filters1, filters2 = filters
        if backend.image_data_format() == 'channels_last':
            bn_axis = 3
        else:
            bn_axis = 1
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        x = Conv2D(filters1, kernel_size,
                   padding='same', use_bias=False,
                   kernel_initializer='he_normal',
                   kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
                   name=conv_name_base + '2a')(input_tensor, params=params)

        x = BatchNormalization(axis=bn_axis, momentum=BATCH_NORM_DECAY, epsilon=BATCH_NORM_EPSILON,
                               name=bn_name_base + '2a')(x, params=params, training=training)
        x = layers.Activation('relu')(x)

        x = layers.Dropout(self.droprate, name=conv_name_base + 'dropout2a')(x)

        x = Conv2D(filters2, kernel_size,
                   padding='same', use_bias=False,
                   kernel_initializer='he_normal',
                   kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
                   name=conv_name_base + '2b')(x, params=params)

        x = BatchNormalization(axis=bn_axis, momentum=BATCH_NORM_DECAY, epsilon=BATCH_NORM_EPSILON,
                               name=bn_name_base + '2b')(x, params=params, training=training)

        x = layers.add([x, input_tensor])
        x = layers.Activation('relu')(x)
        return x

    def conv_building_block(self, input_tensor, params,
                            kernel_size,
                            filters,
                            stage,
                            block,
                            strides=(2, 2),
                            training=None):
        """A block that has a conv layer at shortcut.

        Arguments:
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: current block label, used for generating layer names
        strides: Strides for the first conv layer in the block.
        training: Only used if training keras model with Estimator.  In other
          scenarios it is handled automatically.

        Returns:
        Output tensor for the block.

        Note that from stage 3,
        the first conv layer at main path is with strides=(2, 2)
        And the shortcut should have strides=(2, 2) as well

        """
        filters1, filters2 = filters
        if tf.keras.backend.image_data_format() == 'channels_last':
            bn_axis = 3
        else:
            bn_axis = 1
        conv_name_base = 'res' + str(stage) + block + '_branch'
        bn_name_base = 'bn' + str(stage) + block + '_branch'

        x = Conv2D(filters1, kernel_size, strides=strides,
                   padding='same', use_bias=False,
                   kernel_initializer='he_normal',
                   kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
                   name=conv_name_base + '2a')(input_tensor, params=params)

        x = BatchNormalization(axis=bn_axis, momentum=BATCH_NORM_DECAY, epsilon=BATCH_NORM_EPSILON,
                               name=bn_name_base + '2a')(x, params=params, training=training)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(self.droprate, name=conv_name_base + 'dropout2a')(x)

        x = Conv2D(filters2, kernel_size, padding='same', use_bias=False,
                   kernel_initializer='he_normal',
                   kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
                   name=conv_name_base + '2b')(x, params=params)

        x = BatchNormalization(axis=bn_axis, momentum=BATCH_NORM_DECAY, epsilon=BATCH_NORM_EPSILON,
                               name=bn_name_base + '2b')(x, params=params, training=training)

        shortcut = Conv2D(filters2, (1, 1), strides=strides, use_bias=False,
                          kernel_initializer='he_normal',
                          kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
                          name=conv_name_base + '1')(input_tensor, params=params)

        shortcut = BatchNormalization(axis=bn_axis, momentum=BATCH_NORM_DECAY, epsilon=BATCH_NORM_EPSILON,
                                      name=bn_name_base + '1')(shortcut, params=params,
                                                               training=training)
        x = layers.add([x, shortcut])
        x = layers.Activation('relu')(x)
        return x

    def resnet_block(self, input_tensor, params,
                     size,
                     kernel_size,
                     filters,
                     stage,
                     conv_strides=(2, 2),
                     training=None):
        """A block which applies conv followed by multiple identity blocks.

        Arguments:
        input_tensor: input tensor
        size: integer, number of constituent conv/identity building blocks.
        A conv block is applied once, followed by (size - 1) identity blocks.
        kernel_size: default 3, the kernel size of
            middle conv layer at main path
        filters: list of integers, the filters of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        conv_strides: Strides for the first conv layer in the block.
        training: Only used if training keras model with Estimator.  In other
          scenarios it is handled automatically.

        Returns:
        Output tensor after applying conv and identity blocks.

        """
        x = self.conv_building_block(input_tensor, params, kernel_size, filters, stage=stage,
                                     strides=conv_strides, block='block_0',
                                     training=training)
        for i in range(size - 1):
            x = self.identity_building_block(x, params, kernel_size, filters, stage=stage,
                                             block='block_%d' % (i + 1), training=training)
        return x

    def resnet(self, num_blocks, params, img_input=None, classes=10, training=None):
        """Instantiates the ResNet architecture.

        Arguments:
        num_blocks: integer, the number of conv/identity blocks in each block.
          The ResNet contains 3 blocks with each block containing one conv block
          followed by (layers_per_block - 1) number of idenity blocks. Each
          conv/idenity block has 2 convolutional layers. With the input
          convolutional layer and the pooling layer towards the end, this brings
          the total size of the network to (6*num_blocks + 2)
        classes: optional number of classes to classify images into
        training: Only used if training keras model with Estimator.  In other
        scenarios it is handled automatically.

        Returns:
        A Keras model instance if img_input is None
        Output tensor if img_input is not None

        """
        input_shape = (32, 32, 3)
        if img_input is None:
            img_input = layers.Input(shape=input_shape)
            return_model = True
        else:
            assert img_input.shape[1:] == input_shape
            return_model = False

        if backend.image_data_format() == 'channels_first':
            x = layers.Lambda(lambda x: backend.permute_dimensions(x, (0, 3, 1, 2)),
                              name='transpose')(img_input)
            bn_axis = 1
        else:  # channel_last
            x = img_input
            bn_axis = 3

        x = layers.ZeroPadding2D(padding=(1, 1), name='conv1_pad')(x)

        x = Conv2D(16, (3, 3),
                   strides=(1, 1),
                   padding='valid', use_bias=False,
                   kernel_initializer='he_normal',
                   kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
                   name='conv1')(x, params=params)

        x = BatchNormalization(axis=bn_axis,
                               momentum=BATCH_NORM_DECAY,
                               epsilon=BATCH_NORM_EPSILON,
                               name='bn_conv1',)(x, params=params, training=training)
        x = layers.Activation('relu')(x)

        x = self.resnet_block(x, params, size=num_blocks, kernel_size=3, filters=[16, 16],
                              stage=2, conv_strides=(1, 1), training=training)

        x = self.resnet_block(x, params, size=num_blocks, kernel_size=3, filters=[32, 32],
                              stage=3, conv_strides=(2, 2), training=training)

        x_features = self.resnet_block(x, params, size=num_blocks, kernel_size=3, filters=[64, 64],
                                       stage=4, conv_strides=(2, 2), training=training)

        rm_axes = [1, 2] if backend.image_data_format() == 'channels_last' else [2, 3]
        x_features = layers.Lambda(lambda x: backend.mean(x, rm_axes), name='reduce_mean')(x_features)
        x = Dense(classes,
                  activation=None,
                  kernel_initializer=initializers.RandomNormal(stddev=0.01),
                  kernel_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
                  bias_regularizer=regularizers.l2(L2_WEIGHT_DECAY),
                  name='fc10')(x_features, params=params)

        if return_model:
            inputs = img_input
            # Create model.
            model = tf.keras.models.Model(inputs, x, name='resnet')
            return model
        else:
            return x, x_features
