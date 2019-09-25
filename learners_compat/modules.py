"""Custom modules to call tf layers with external weights."""

import tensorflow as tf
from tensorflow.python.keras import backend
from tensorflow.python.keras import initializers
from tensorflow.python.keras import layers
from tensorflow.python.keras import regularizers
from tensorflow.python.keras import constraints
from tensorflow.python.framework import ops
from tensorflow.python.framework import common_shapes
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import array_ops
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.training import distribution_strategy_context
from tensorflow.python.keras.utils import conv_utils

from tensorflow.python.keras.engine.base_layer import InputSpec
from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util.tf_export import tf_export


class Conv2D(layers.Conv2D):
    """As Conv2D with overwritten call method."""

    __doc__ += layers.Conv2D.__doc__

    def call(self, inputs, params=None):

        if params[self.name + '/kernel:0'] is None:
            return super(layers.Conv2D, self).call(inputs)
        else:
            kernel = params.get(self.name + '/kernel:0')
            bias = params.get(self.name + '/bias:0')
        outputs = self._convolution_op(inputs, kernel)

        if self.use_bias:
            if self.data_format == 'channels_first':
                if self.rank == 1:
                    # nn.bias_add does not accept a 1D input tensor.
                    bias = array_ops.reshape(bias, (1, self.filters, 1))
                    outputs += bias
                else:
                    outputs = nn.bias_add(outputs, bias, data_format='NCHW')
            else:
                outputs = nn.bias_add(outputs, bias, data_format='NHWC')

        if self.activation is not None:
            return self.activation(outputs)
        return outputs


class BatchNormalization(layers.BatchNormalization):
    """As Batchnorm (v2) with overwritten call method."""

    __doc__ += layers.BatchNormalization.__doc__

    def call(self, inputs, params=None, training=None):

        if params[self.name + '/gamma:0'] is None:
            return super(layers.BatchNormalization, self).call(inputs)
        else:
            gamma = params.get(self.name + '/gamma:0')
            beta = params.get(self.name + '/beta:0')

        original_training_value = training
        if training is None:
            training = backend.learning_phase()

        # Compute the axes along which to reduce the mean / variance
        input_shape = inputs.get_shape()
        ndims = len(input_shape)
        reduction_axes = [i for i in range(ndims) if i not in self.axis]

        # Broadcasting only necessary for single-axis batch norm where the axis is
        # not the last dimension
        broadcast_shape = [1] * ndims
        broadcast_shape[self.axis[0]] = input_shape[self.axis[0]].value

        def _broadcast(v):
            if (v is not None and len(v.get_shape()) != ndims and reduction_axes != list(range(ndims - 1))):
                return array_ops.reshape(v, broadcast_shape)
            return v

        scale, offset = _broadcast(gamma), _broadcast(beta)

        def _compose_transforms(scale, offset, then_scale, then_offset):
            if then_scale is not None:
                scale *= then_scale
                offset *= then_scale
            if then_offset is not None:
                offset += then_offset
            return (scale, offset)

        # Determine a boolean value for `training`: could be True, False, or None.
        training_value = tf_utils.constant_value(training)
        if training_value is not False:
            if self.adjustment:
                adj_scale, adj_bias = self.adjustment(array_ops.shape(inputs))
                # Adjust only during training.
                adj_scale = tf_utils.smart_cond(training,
                                                lambda: adj_scale,
                                                lambda: array_ops.ones_like(adj_scale))
                adj_bias = tf_utils.smart_cond(training,
                                               lambda: adj_bias,
                                               lambda: array_ops.zeros_like(adj_bias))
                scale, offset = _compose_transforms(adj_scale, adj_bias, scale, offset)

        # Some of the computations here are not necessary when training==False
        # but not a constant. However, this makes the code simpler.
        keep_dims = self.virtual_batch_size is not None or len(self.axis) > 1
        mean, variance = nn.moments(inputs, reduction_axes, keep_dims=keep_dims)

        moving_mean = self.moving_mean
        moving_variance = self.moving_variance

        mean = tf_utils.smart_cond(training,
                                   lambda: mean,
                                   lambda: moving_mean)
        variance = tf_utils.smart_cond(training,
                                       lambda: variance,
                                       lambda: moving_variance)

        if self.virtual_batch_size is not None:
            # This isn't strictly correct since in ghost batch norm, you are
            # supposed to sequentially update the moving_mean and moving_variance
            # with each sub-batch. However, since the moving statistics are only
            # used during evaluation, it is more efficient to just update in one
            # step and should not make a significant difference in the result.
            new_mean = math_ops.reduce_mean(mean, axis=1, keepdims=True)
            new_variance = math_ops.reduce_mean(variance, axis=1, keepdims=True)
        else:
            new_mean, new_variance = mean, variance

        if self.renorm:
            r, d, new_mean, new_variance = self._renorm_correction_and_moments(
                new_mean, new_variance, training)
            # When training, the normalized values (say, x) will be transformed as
            # x * gamma + beta without renorm, and (x * r + d) * gamma + beta
            # = x * (r * gamma) + (d * gamma + beta) with renorm.
            r = _broadcast(array_ops.stop_gradient(r, name='renorm_r'))
            d = _broadcast(array_ops.stop_gradient(d, name='renorm_d'))
            scale, offset = _compose_transforms(r, d, scale, offset)

        def _do_update(var, value):
            return self._assign_moving_average(var, value, self.momentum)

        mean_update = tf_utils.smart_cond(
            training,
            lambda: _do_update(self.moving_mean, new_mean),
            lambda: self.moving_mean)
        variance_update = tf_utils.smart_cond(
            training,
            lambda: _do_update(self.moving_variance, new_variance),
            lambda: self.moving_variance)
        self.add_update(mean_update, inputs=True)
        self.add_update(variance_update, inputs=True)
        # mean, variance = self.moving_mean, self.moving_variance

        mean = math_ops.cast(mean, inputs.dtype)
        variance = math_ops.cast(variance, inputs.dtype)
        if offset is not None:
            offset = math_ops.cast(offset, inputs.dtype)
        outputs = nn.batch_normalization(inputs,
                                         _broadcast(mean),
                                         _broadcast(variance),
                                         offset,
                                         scale,
                                         self.epsilon)
        # If some components of the shape got lost due to adjustments, fix that.
        outputs.set_shape(input_shape)

        if self.virtual_batch_size is not None:
            outputs = undo_virtual_batching(outputs)
        if original_training_value is None:
            outputs._uses_learning_phase = True  # pylint: disable=protected-access
        return outputs


class Dense(layers.Dense):
    """As normal Dense with overwritten call method."""

    __doc__ += layers.Dense.__doc__

    def call(self, inputs, params=None):

        if params[self.name + '/kernel:0'] is None:
            return super(layers.Dense, self).call(inputs)
        else:
            kernel = params.get(self.name + '/kernel:0')
            bias = params.get(self.name + '/bias:0')

        inputs = ops.convert_to_tensor(inputs, dtype=self.dtype)
        rank = common_shapes.rank(inputs)
        if rank > 2:
            # Broadcasting is required for the inputs.
            outputs = standard_ops.tensordot(inputs, kernel, [[rank - 1], [0]])
            # Reshape the output back to the original ndim of the input.
            if not context.executing_eagerly():
                shape = inputs.get_shape().as_list()
                output_shape = shape[:-1] + [self.units]
                outputs.set_shape(output_shape)
        else:
            outputs = gen_math_ops.mat_mul(inputs, kernel)
        if self.use_bias:
            outputs = nn.bias_add(outputs, bias)
        if self.activation is not None:
            return self.activation(outputs)  # pylint: disable=not-callable
        return outputs


class DepthwiseConv2D(layers.DepthwiseConv2D):
    """As normal DepthwiseConv2D but with overwritten call method."""

    __doc__ += layers.DepthwiseConv2D.__doc__

    def call(self, inputs, params=None):
        if params[self.name + '/depthwise_kernel:0'] is None:
            return super(layers.DepthwiseConv2D, self).call(inputs)
        else:
            depthwise_kernel = params.get(self.name + '/depthwise_kernel:0')
            bias = params.get(self.name + '/bias:0')

        outputs = backend.depthwise_conv2d(
            inputs,
            depthwise_kernel,
            strides=self.strides,
            padding=self.padding,
            dilation_rate=self.dilation_rate,
            data_format=self.data_format)

        if self.use_bias:
            outputs = backend.bias_add(
                outputs,
                bias,
                data_format=self.data_format)

        if self.activation is not None:
            return self.activation(outputs)

        return outputs


class SeparableConv2D(layers.SeparableConv2D):
    """As separable Conv2D but with overwritten call to allow for external weights."""

    __doc__ += layers.SeparableConv2D.__doc__

    def call(self, inputs, params=None):
        if params[self.name + '/depthwise_kernel:0'] is None:
            return super(layers.SeparableConv2D, self).call(inputs)
        else:
            depthwise_kernel = params.get(self.name + '/depthwise_kernel:0')
            pointwise_kernel = params.get(self.name + '/pointwise_kernel:0')
            bias = params.get(self.name + '/bias:0')
        # Apply the actual ops.
        if self.data_format == 'channels_last':
            strides = (1,) + self.strides + (1,)
        else:
            strides = (1, 1) + self.strides
        outputs = nn.separable_conv2d(
            inputs,
            depthwise_kernel,
            pointwise_kernel,
            strides=strides,
            padding=self.padding.upper(),
            rate=self.dilation_rate,
            data_format=conv_utils.convert_data_format(self.data_format, ndim=4))

        if self.use_bias:
            outputs = nn.bias_add(
                outputs,
                bias,
                data_format=conv_utils.convert_data_format(self.data_format, ndim=4))

        if self.activation is not None:
            return self.activation(outputs)
        return outputs
