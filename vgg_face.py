"""
Code mainly taken from https://github.com/ZZUTK/Tensorflow-VGG-face
"""

import tensorflow as tf
import numpy as np
from scipy.io import loadmat

VGG_FACE_MEAN = np.array([129.1862793, 104.76238251, 93.59396362])

# TODO: make the net trainable.
def vgg_face(param_path, input_maps):
    input_shape = input_maps.get_shape().as_list()
    if len(input_shape) != 4 or input_shape[1] != 224 or input_shape[2] != 224:
        raise AssertionError("Input shape has to be 4 dimensional with height and with = 224")

    data = loadmat(param_path)

    # read meta info
    meta = data['meta']
    classes = meta['classes']
    class_names = classes[0][0]['description'][0][0]
    normalization = meta['normalization']
    average_image = np.squeeze(normalization[0][0]['averageImage'][0][0][0][0])
    image_size = np.squeeze(normalization[0][0]['imageSize'][0][0])
    input_maps = tf.image.resize_images(input_maps, (image_size[0], image_size[1]))

    # read layer info
    layers = data['layers']
    current = input_maps
    network = {}
    for layer in layers[0]:
        name = layer[0]['name'][0][0]
        layer_type = layer[0]['type'][0][0]
        if layer_type == 'conv':
            if name[:2] == 'fc':
                padding = 'VALID'
            else:
                padding = 'SAME'
            stride = layer[0]['stride'][0][0]
            kernel, bias = layer[0]['weights'][0][0]
            # kernel = np.transpose(kernel, (1, 0, 2, 3))
            bias = np.squeeze(bias).reshape(-1)
            conv = tf.nn.conv2d(current, tf.constant(kernel),
                                strides=(1, stride[0], stride[0], 1), padding=padding)
            current = tf.nn.bias_add(conv, bias)
            print name, 'stride:', stride, 'kernel size:', np.shape(kernel)
        elif layer_type == 'relu':
            current = tf.nn.relu(current)
            print name
        elif layer_type == 'pool':
            stride = layer[0]['stride'][0][0]
            pool = layer[0]['pool'][0][0]
            current = tf.nn.max_pool(current, ksize=(1, pool[0], pool[1], 1),
                                     strides=(1, stride[0], stride[0], 1), padding='SAME')
            print name, 'stride:', stride
        elif layer_type == 'softmax':
            current = tf.nn.softmax(tf.reshape(current, [-1, len(class_names)]))
            print name

        network[name] = current

    return network, average_image, class_names


def vgg_face_trainable(param_path, input_maps, reuse = False):
    with tf.variable_scope('vgg_face', reuse=reuse):
        data = loadmat(param_path)

        # read meta info
        meta = data['meta']
        classes = meta['classes']
        class_names = classes[0][0]['description'][0][0]
        normalization = meta['normalization']
        average_image = np.squeeze(normalization[0][0]['averageImage'][0][0][0][0])
        image_size = np.squeeze(normalization[0][0]['imageSize'][0][0])
        input_maps = tf.image.resize_images(input_maps, (image_size[0], image_size[1]))

        # read layer info
        layers = data['layers']
        current = input_maps
        network = {}
        for layer in layers[0]:
            name = layer[0]['name'][0][0]
            layer_type = layer[0]['type'][0][0]
            if layer_type == 'conv':
                if name[:2] == 'fc':
                    padding = 'VALID'
                else:
                    padding = 'SAME'
                stride = layer[0]['stride'][0][0]
                kernel, bias = layer[0]['weights'][0][0]
                bias = np.squeeze(bias).reshape(-1)
                kernel = tf.get_variable(name=name+'_kernel',initializer=kernel)
                bias = tf.get_variable(name=name+'_bias',initializer=bias)
                # kernel = np.transpose(kernel, (1, 0, 2, 3))
                # conv = tf.nn.conv2d(current, tf.constant(kernel),
                #                     strides=(1, stride[0], stride[0], 1), padding=padding)
                conv = tf.nn.conv2d(current, kernel,
                                    strides=(1, stride[0], stride[0], 1), padding=padding)
                current = tf.nn.bias_add(conv, bias)
                print name, 'stride:', stride, 'kernel size:', np.shape(kernel), 'output_shape:', str(current.get_shape().as_list())
            elif layer_type == 'relu':
                current = tf.nn.relu(current)
                print name
            elif layer_type == 'pool':
                stride = layer[0]['stride'][0][0]
                pool = layer[0]['pool'][0][0]
                current = tf.nn.max_pool(current, ksize=(1, pool[0], pool[1], 1),
                                         strides=(1, stride[0], stride[0], 1), padding='SAME')
                print name, 'stride:', stride
            elif layer_type == 'softmax':
                current = tf.nn.softmax(tf.reshape(current, [-1, len(class_names)]))
                print name

            network[name] = current

        return network, average_image, class_names
