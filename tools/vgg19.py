
import tensorflow as tf

import numpy as np
import time
import sys

VGG_MEAN = [103.939, 116.779, 123.68]


class Vgg19(tf.Module):
    def __init__(self, vgg19_npy_path='./vgg19_weight/vgg19_no_fc.npy', name='vgg19'):
        super().__init__(name=name)
        if vgg19_npy_path is not None:
            self.data_dict = np.load(
                vgg19_npy_path, encoding='latin1', allow_pickle=True).item()
            print("npy file loaded ------- ", vgg19_npy_path)
        else:
            self.data_dict = None
            print("npy file load error!")
            sys.exit(1)

        # Pre-build constant tensors for all layers to avoid recreating them
        self._filters = {}
        self._biases = {}
        for name_key in self.data_dict:
            self._filters[name_key] = tf.constant(self.data_dict[name_key][0], name=f"{name_key}_filter")
            self._biases[name_key] = tf.constant(self.data_dict[name_key][1], name=f"{name_key}_biases")

    def _preprocess(self, rgb):
        """Convert RGB [-1,1] to BGR with VGG mean subtracted."""
        rgb_scaled = ((rgb + 1) / 2) * 255.0  # [-1, 1] ~ [0, 255]
        red, green, blue = tf.split(
            axis=3, num_or_size_splits=3, value=rgb_scaled)
        bgr = tf.concat(axis=3, values=[blue - VGG_MEAN[0],
                                        green - VGG_MEAN[1],
                                        red - VGG_MEAN[2]])
        return bgr

    def build(self, rgb):
        """
        Build VGG19 up to conv4_4 (no activation).
        input format: rgb image with shape [batch_size, h, w, 3]
        scale: (-1, 1)
        """
        bgr = self._preprocess(rgb)

        conv1_1 = self.conv_layer(bgr, "conv1_1")
        conv1_2 = self.conv_layer(conv1_1, "conv1_2")
        pool1 = self.max_pool(conv1_2)

        conv2_1 = self.conv_layer(pool1, "conv2_1")
        conv2_2 = self.conv_layer(conv2_1, "conv2_2")
        pool2 = self.max_pool(conv2_2)

        conv3_1 = self.conv_layer(pool2, "conv3_1")
        conv3_2 = self.conv_layer(conv3_1, "conv3_2")
        conv3_3 = self.conv_layer(conv3_2, "conv3_3")
        conv3_4 = self.conv_layer(conv3_3, "conv3_4")
        pool3 = self.max_pool(conv3_4)

        conv4_1 = self.conv_layer(pool3, "conv4_1")
        conv4_2 = self.conv_layer(conv4_1, "conv4_2")
        conv4_3 = self.conv_layer(conv4_2, "conv4_3")

        conv4_4_no_activation = self.no_activation_conv_layer(conv4_3, "conv4_4")

        return conv4_4_no_activation

    def build_multi(self, rgb):
        """
        Build VGG19 and return features from conv4_4, conv3_3, conv2_2 (all no activation).
        input format: rgb image with shape [batch_size, h, w, 3]
        scale: (-1, 1)
        """
        bgr = self._preprocess(rgb)

        conv1_1 = self.conv_layer(bgr, "conv1_1")
        conv1_2 = self.conv_layer(conv1_1, "conv1_2")
        pool1 = self.max_pool(conv1_2)

        conv2_1 = self.conv_layer(pool1, "conv2_1")
        conv2_2 = self.conv_layer(conv2_1, "conv2_2")
        conv2_2_no_activation = self.no_activation_conv_layer(conv2_1, "conv2_2")
        pool2 = self.max_pool(conv2_2)

        conv3_1 = self.conv_layer(pool2, "conv3_1")
        conv3_2 = self.conv_layer(conv3_1, "conv3_2")
        conv3_3_no_activation = self.no_activation_conv_layer(conv3_2, "conv3_3")
        conv3_3 = self.conv_layer(conv3_2, "conv3_3")
        conv3_4 = self.conv_layer(conv3_3, "conv3_4")
        pool3 = self.max_pool(conv3_4)

        conv4_1 = self.conv_layer(pool3, "conv4_1")
        conv4_2 = self.conv_layer(conv4_1, "conv4_2")
        conv4_3 = self.conv_layer(conv4_2, "conv4_3")
        conv4_4_no_activation = self.no_activation_conv_layer(conv4_3, "conv4_4")

        return conv4_4_no_activation, conv3_3_no_activation, conv2_2_no_activation

    def max_pool(self, bottom):
        return tf.nn.max_pool2d(bottom, ksize=2, strides=2, padding='SAME')

    def conv_layer(self, bottom, name):
        filt = self._filters[name]
        conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')
        bias = tf.nn.bias_add(conv, self._biases[name])
        relu = tf.nn.relu(bias)
        return relu

    def no_activation_conv_layer(self, bottom, name):
        filt = self._filters[name]
        conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')
        x = tf.nn.bias_add(conv, self._biases[name])
        return x
