import tensorflow as tf
import numpy as np

#gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8)
#tf.enable_eager_execution(config=tf.ConfigProto(gpu_options=gpu_options))
tf.enable_eager_execution()

def conv2d(x, num_filters, filter_size=(3, 3), stride=(1, 1), pad="SAME",
           dtype=tf.float32, collections=None,
           summary_tag=None):
    stride_shape = [1, stride[0], stride[1], 1]
    filter_shape = [filter_size[0], filter_size[1], int(x.get_shape()[3]), num_filters]

    # there are "num input feature maps * filter height * filter width"
    # inputs to each hidden unit
    fan_in = intprod(filter_shape[:3])
    # each unit in the lower layer receives a gradient from:
    # "num output feature maps * filter height * filter width" /
    #   pooling size
    fan_out = intprod(filter_shape[:2]) * num_filters
    # initialize weights with random weights
    w_bound = np.sqrt(6. / (fan_in + fan_out))

    w = tf.get_variable("W", filter_shape, dtype,
                        tf.random_uniform_initializer(-w_bound, w_bound),
                        collections=collections)
    b = tf.get_variable("b", [1, 1, 1, num_filters], initializer=tf.zeros_initializer(),
                        collections=collections)

    if summary_tag is not None:
        tf.summary.image(summary_tag,
                         tf.transpose(
                             tf.reshape(w, [filter_size[0], filter_size[1], -1, 1]),
                             [2, 0, 1, 3]),
                         max_images=10)

    return tf.nn.conv2d(x, w, stride_shape, pad) + b


def flattenallbut0(x):
    return tf.reshape(x, [-1, intprod(x.get_shape().as_list()[1:])])


def normc_initializer(std=1.0, axis=0):
    def _initializer(shape, dtype=None, partition_info=None):  # pylint: disable=W0613
        out = np.random.randn(*shape).astype(dtype.as_numpy_dtype)
        out *= std / np.sqrt(np.square(out).sum(axis=axis, keepdims=True))
        return tf.constant(out)

    return _initializer

def intprod(x):
    return int(np.prod(x))
