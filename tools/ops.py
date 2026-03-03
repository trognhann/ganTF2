import tensorflow.compat.v1 as tf
import os
from .tf_color_ops import rgb_to_lab
from .vgg19 import Vgg19


# Xavier : tf.initializers.glorot_uniform()
# He : tf.initializers.variance_scaling()
# Normal : tf.random_normal_initializer(mean=0.0, stddev=0.02)
# l2_decay : tf.keras.regularizers.l2(0.0001)


weight_init = tf.initializers.glorot_uniform()


def l2_regularizer(weight=0.0001):
    def l2(x):
        return weight * tf.nn.l2_loss(x)
    return l2


weight_regularizer = l2_regularizer(0.0001)


##################################################################################
# Activation function
##################################################################################

def lrelu(x, alpha=0.2):
    return tf.nn.leaky_relu(x, alpha)


def relu(x):
    return tf.nn.relu(x)


def tanh(x):
    return tf.tanh(x)


def sigmoid(x):
    return tf.sigmoid(x)


def h_swish(x):
    return x * tf.nn.relu6(x+3)/6.0


##################################################################################
# Normalization function
##################################################################################

def GroupNorm(x, G=16, eps=1e-5):
    # x: input features with shape [N, H, W, C]
    # gamma, beta: scale and offset, with shape [1, 1, 1, C]
    # G: number of groups or GN
    N, H, W, C = x.shape
    x = tf.reshape(x, [N, G, H, W, C // G])

    gamma = tf.Variable(tf.ones([x.get_shape()[-1]]), trainable=True)
    beta = tf.Variable(tf.zeros([x.get_shape()[-1]]), trainable=True)

    mean, var = tf.nn.moments(x, [2, 3, 4], keep_dims=True)
    x = (x - mean) / tf.sqrt(var + eps)

    x = tf.reshape(x, [N, H, W, C])

    return x * gamma + beta


def instance_norm(x, scope=None):
    with tf.variable_scope(scope, default_name="instance_norm"):
        depth = x.get_shape()[-1]
        scale = tf.get_variable(
            "scale", [depth], initializer=tf.ones_initializer())
        offset = tf.get_variable(
            "offset", [depth], initializer=tf.zeros_initializer())
        mean, variance = tf.nn.moments(x, [1, 2], keep_dims=True)
        epsilon = 1e-5
        inv = tf.rsqrt(variance + epsilon)
        normalized = (x - mean) * inv
        return scale * normalized + offset


def layer_norm(x, scope='layer_norm'):
    with tf.variable_scope(scope):
        depth = x.get_shape()[-1]
        scale = tf.get_variable(
            "scale", [depth], initializer=tf.ones_initializer())
        offset = tf.get_variable(
            "offset", [depth], initializer=tf.zeros_initializer())
        mean, variance = tf.nn.moments(x, [1, 2, 3], keep_dims=True)
        epsilon = 1e-5
        inv = tf.rsqrt(variance + epsilon)
        normalized = (x - mean) * inv
        return scale * normalized + offset


def batch_norm(x, is_training=True, scope='batch_norm'):
    with tf.variable_scope(scope):
        return batch_norm_wrapper(x, is_training)


def batch_norm_wrapper(inputs, is_training, decay=0.999, epsilon=0.001):
    scale = tf.Variable(tf.ones([inputs.get_shape()[-1]]))
    beta = tf.Variable(tf.zeros([inputs.get_shape()[-1]]))
    pop_mean = tf.Variable(tf.zeros([inputs.get_shape()[-1]]), trainable=False)
    pop_var = tf.Variable(tf.ones([inputs.get_shape()[-1]]), trainable=False)
    if is_training:
        batch_mean, batch_var = tf.nn.moments(inputs, [0, 1, 2])
        train_mean = tf.assign(pop_mean, pop_mean *
                               decay + batch_mean * (1 - decay))
        train_var = tf.assign(pop_var, pop_var * decay +
                              batch_var * (1 - decay))
        with tf.control_dependencies([train_mean, train_var]):
            return tf.nn.batch_normalization(inputs, batch_mean, batch_var, beta, scale, epsilon)
    else:
        return tf.nn.batch_normalization(inputs, pop_mean, pop_var, beta, scale, epsilon)


def spectral_norm(w, iteration=1):
    w_shape = w.shape.as_list()
    w = tf.reshape(w, [-1, w_shape[-1]])

    u = tf.get_variable(
        "u", [1, w_shape[-1]], initializer=tf.truncated_normal_initializer(), trainable=False)

    u_hat = u
    v_hat = None
    for i in range(iteration):
        """
        power iteration
        Usually iteration = 1 will be enough
        """
        v_ = tf.matmul(u_hat, tf.transpose(w))
        v_hat = l2_norm(v_)

        u_ = tf.matmul(v_hat, w)
        u_hat = l2_norm(u_)

    sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))
    w_norm = w / sigma

    with tf.control_dependencies([u.assign(u_hat)]):
        w_norm = tf.reshape(w_norm, w_shape)

    return w_norm


def l2_norm(v, eps=1e-12):
    return v / (tf.reduce_sum(v ** 2) ** 0.5 + eps)

##################################################################################
# Layer
##################################################################################


def conv(x, channels, kernel=4, stride=2, sn=False, pad_type='reflect', use_bias=False, scope='conv_0'):
    with tf.variable_scope(scope):
        if (kernel - stride) % 2 == 0:
            pad = (kernel - stride) // 2
            pad_top, pad_bottom, pad_left, pad_right = pad, pad, pad, pad

        else:
            pad = (kernel - stride) // 2
            pad_bottom, pad_right = pad, pad,
            pad_top, pad_left = kernel - stride - pad_bottom, kernel - stride - pad_right

        if pad_type == 'zero':
            x = tf.pad(x, [[0, 0], [pad_top, pad_bottom],
                       [pad_left, pad_right], [0, 0]])
        if pad_type == 'reflect':
            x = tf.pad(x, [[0, 0], [pad_top, pad_bottom], [
                       pad_left, pad_right], [0, 0]], mode='REFLECT')

        if sn:
            w = tf.get_variable("kernel", shape=[kernel, kernel, x.get_shape(
            )[-1], channels], initializer=weight_init, regularizer=weight_regularizer)
            x = tf.nn.conv2d(input=x, filter=spectral_norm(w), strides=[
                             1, stride, stride, 1], padding='VALID')
            if use_bias:
                bias = tf.get_variable(
                    "bias", [channels], initializer=tf.constant_initializer(0.0))
                x = tf.nn.bias_add(x, bias)

        else:
            w = tf.get_variable("kernel", shape=[kernel, kernel, x.get_shape(
            )[-1], channels], initializer=weight_init, regularizer=weight_regularizer)
            x = tf.nn.conv2d(input=x, filter=w, strides=[
                             1, stride, stride, 1], padding='VALID')
            if use_bias:
                bias = tf.get_variable(
                    "bias", [channels], initializer=tf.constant_initializer(0.0))
                x = tf.nn.bias_add(x, bias)
        return x


def Conv2D(inputs, filters, kernel_size=3, strides=1, padding='VALID', Use_bias=None, activation_fn=None, scope='conv2d'):
    if (kernel_size - strides) % 2 == 0:
        pad = (kernel_size - strides) // 2
        pad_top, pad_bottom, pad_left, pad_right = pad, pad, pad, pad
    else:
        pad = (kernel_size - strides) // 2
        pad_bottom, pad_right = pad, pad,
        pad_top, pad_left = kernel_size - strides - \
            pad_bottom,  kernel_size - strides - pad_right

    inputs = tf.pad(inputs, [[0, 0], [pad_top, pad_bottom], [
                    pad_left, pad_right], [0, 0]], mode="REFLECT")
    with tf.variable_scope(scope):
        w = tf.get_variable("kernel", shape=[kernel_size, kernel_size, inputs.get_shape(
        )[-1], filters], initializer=weight_init, regularizer=weight_regularizer)
        x = tf.nn.conv2d(input=inputs, filter=w, strides=[
                         1, strides, strides, 1], padding=padding)
        if Use_bias is not None:
            bias = tf.get_variable("bias", [
                                   filters], initializer=Use_bias if Use_bias is not None else tf.zeros_initializer())
            x = tf.nn.bias_add(x, bias)
        if activation_fn:
            x = activation_fn(x)
        return x


def Conv2d_LN_LReLU(inputs, filters, kernel_size=3, strides=1, name=None, padding='VALID', Use_bias=None):
    x = Conv2D(inputs, filters, kernel_size, strides,
               padding=padding, Use_bias=Use_bias)
    x = layer_norm(x, scope=name)
    return lrelu(x)


def Conv2d_IN_LReLU(inputs, filters, kernel_size=3, strides=1, name=None, padding='VALID', Use_bias=None):
    x = Conv2D(inputs, filters, kernel_size, strides,
               padding=padding, Use_bias=Use_bias)
    x = instance_norm(x, scope=name)
    return lrelu(x)


##################################################################################
# Sampling
##################################################################################

def flatten(x):
    return tf.compat.v1.layers.flatten(x)


def global_avg_pooling(x, keepdims=True):
    gap = tf.reduce_mean(x, axis=[1, 2], keepdims=keepdims)
    return gap


def global_max_pooling(x, keepdims=True):
    gmp = tf.reduce_max(x, axis=[1, 2], keepdims=keepdims)
    return gmp


"""Attention"""


def External_attention_v3(x, is_training, k=128, scope='External_attention'):
    idn = x
    b, h, w, c = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], x.shape[-1]
    with tf.variable_scope(scope):
        w_kernel = tf.get_variable("kernel", [
                                   1, c, k], tf.float32, initializer=weight_init, regularizer=weight_regularizer)
        x = Conv2D(x, c, 1, 1, scope='conv1')
        x = tf.reshape(x, shape=[b, -1, c])
        attn = tf.nn.conv1d(x, w_kernel, stride=1, padding='VALID')
        attn = tf.nn.softmax(attn, axis=1)
        attn = attn / (1e-9 + tf.reduce_sum(attn, axis=2, keepdims=True))
    # with tf.variable_scope(scope, reuse=True):
        w_kernel = tf.transpose(w_kernel, perm=[0, 2, 1])
        x = tf.nn.conv1d(attn, w_kernel, stride=1, padding='VALID')
        x = tf.reshape(x, [b, h, w, c])
        x = Conv2D(x, c, 1, 1, scope='conv2')
        x = batch_norm(x, is_training, scope='bn')
        # x = LADE(x)
        # x = layer_norm(x)
        # x = instance_norm(x)
        # x = batch_norm(x, is_training)
        x = x + idn
        out = lrelu(x)
    return out


def External_attention(x, is_training, k=64, scope='External_attention'):
    idn = x
    b, h, w, c = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], x.shape[-1]
    with tf.variable_scope(scope):
        w_kernel = tf.get_variable("mk_kernel", [
                                   1, c, k], tf.float32, initializer=weight_init, regularizer=weight_regularizer)
        x = Conv2D(x, c, 1, 1,)
        x = tf.reshape(x, shape=[b, -1, c])
        attn = tf.nn.conv1d(x, w_kernel, 1, 'VALID')
        attn = tf.nn.softmax(attn, axis=1)
        attn = attn / (1e-9 + tf.reduce_sum(attn, axis=2, keepdims=True))

        w_kernel = tf.get_variable("mv_kernel", [
                                   1, k, c], tf.float32, initializer=weight_init, regularizer=weight_regularizer)
        x = tf.nn.conv1d(attn, w_kernel, 1, 'VALID')
        x = tf.reshape(x, [b, h, w, c])
        x = Conv2D(x, c, 1, 1)
        x = batch_norm_wrapper(x, is_training)
        x = x + idn
        out = lrelu(x)
    return out


def LADE_D(x, sn=False, name=''):
    eps = 1e-5
    ch = x.shape[-1]
    tx = conv(x, ch, 1, 1, sn=sn, scope=name+'_conv_IN')
    t_mean, t_sigma = tf.nn.moments(tx, axes=[1, 2], keep_dims=True)
    in_mean, in_sigma = tf.nn.moments(x, axes=[1, 2], keep_dims=True)
    x_in = (x - in_mean) / (tf.sqrt(in_sigma + eps))
    x = x_in * (tf.sqrt(t_sigma + eps)) + t_mean
    return x


def LADE(x, scope='LADE'):
    eps = 1e-5
    ch = x.shape[-1]
    with tf.variable_scope(scope):
        tx = Conv2D(x, ch, 1, 1, scope='conv_IN')
        t_mean, t_sigma = tf.nn.moments(tx, axes=[1, 2], keep_dims=True)
        in_mean, in_sigma = tf.nn.moments(x, axes=[1, 2], keep_dims=True)
        x_in = (x - in_mean) / (tf.sqrt(in_sigma + eps))
        x = x_in * (tf.sqrt(t_sigma + eps)) + t_mean
        return x


def conv_LADE_Lrelu(inputs, filters, kernel_size=3, strides=1, scope='conv_LADE'):
    with tf.variable_scope(scope):
        x = Conv2D(inputs, filters, kernel_size, strides, scope='conv')
        x = LADE(x, scope='LADE')
        return lrelu(x)

##################################################################################
# Loss function
##################################################################################


def L1_loss(x, y):
    loss = tf.reduce_mean(tf.abs(x - y))
    return loss


def L2_loss(x, y):
    loss = tf.reduce_mean(tf.square(x - y))
    return loss


def Huber_loss(x, y, delta=1.0):
    return tf.compat.v1.losses.huber_loss(x, y, delta=delta)


def regularization_loss(scope_name):
    """
    If you want to use "Regularization"
    g_loss += regularization_loss('generator')
    d_loss += regularization_loss('discriminator')
    """
    collection_regularization = tf.get_collection(
        tf.GraphKeys.REGULARIZATION_LOSSES)
    loss = []
    for item in collection_regularization:
        if scope_name in item.name:
            loss.append(item)
    return tf.reduce_sum(loss)


def generator_loss(fake):
    fake_loss = tf.reduce_mean(tf.square(fake - 0.9))
    return fake_loss


def discriminator_loss(anime_logit, fake_logit):
    # lsgan :
    anime_gray_logit_loss = tf.reduce_mean(tf.square(anime_logit - 0.9))
    fake_gray_logit_loss = tf.reduce_mean(tf.square(fake_logit-0.1))
    # loss =   0.5 * anime_gray_logit_loss  \ # Hayao
    loss = 0.5 * anime_gray_logit_loss  \
        + 1.0 * fake_gray_logit_loss
    return loss


def discriminator_loss_346(fake_logit):
    # lsgan :
    fake_logit_loss = tf.reduce_mean(tf.square(fake_logit - 0.1))
    loss = 1.0 * fake_logit_loss
    return loss

# main


def discriminator_loss_m(real, fake):
    real_loss = tf.reduce_mean(tf.square(real - 1.))
    fake_loss = tf.reduce_mean(tf.square(fake))
    loss = real_loss + fake_loss
    return loss


def generator_loss_m(fake):
    loss = tf.reduce_mean(tf.square(fake - 1.))
    return loss


def gram(x):
    shape_x = tf.shape(x)
    b = shape_x[0]
    c = shape_x[3]
    x = tf.reshape(x, [b, -1, c])
    return tf.matmul(tf.transpose(x, [0, 2, 1]), x) / tf.cast((tf.size(x) // b), tf.float32)


"""
vgg19 obj
"""
vgg19 = None


def init_vgg(vgg_dir):
    global vgg19
    if vgg19 is None:
        vgg19_path = os.path.join(vgg_dir, 'vgg19_no_fc.npy')
        vgg19 = Vgg19(vgg19_path)


def VGG_LOSS(x, y):
    # The number of feature channels in layer 4-4 of vgg19 is 512
    x = vgg19.build(x)
    y = vgg19.build(y)
    c = x.get_shape().as_list()[-1]
    return L1_loss(x, y)/tf.cast(c, tf.float32)


def con_loss(real, fake, weight=1.0):
    return weight * VGG_LOSS(real, fake)


def region_smoothing_loss(seg, fake, weight):
    return VGG_LOSS(seg, fake) * weight


def style_loss(style, fake, weight):
    style_feat = vgg19.build(style)
    fake_feat = vgg19.build(fake)
    return weight * L1_loss(gram(style_feat), gram(fake_feat))/tf.cast(style_feat.get_shape().as_list()[-1], tf.float32)


def style_loss_decentralization_3(style, fake, weight):
    style_4, style_3, style_2 = vgg19.build_multi(style)
    fake_4, fake_3, fake_2 = vgg19.build_multi(fake)
    dim = [1, 2]
    style_2 -= tf.reduce_mean(style_2, axis=dim, keep_dims=True)
    fake_2 -= tf.reduce_mean(fake_2, axis=dim, keep_dims=True)
    c_2 = fake_2.get_shape().as_list()[-1]

    style_3 -= tf.reduce_mean(style_3, axis=dim, keep_dims=True)
    fake_3 -= tf.reduce_mean(fake_3, axis=dim, keep_dims=True)
    c_3 = fake_3.get_shape().as_list()[-1]

    style_4 -= tf.reduce_mean(style_4, axis=dim, keep_dims=True)
    fake_4 -= tf.reduce_mean(fake_4, axis=dim, keep_dims=True)
    c_4 = fake_4.get_shape().as_list()[-1]

    loss4_4 = L1_loss(gram(style_4), gram(fake_4))/tf.cast(c_4, tf.float32)
    loss3_3 = L1_loss(gram(style_3), gram(fake_3))/tf.cast(c_3, tf.float32)
    loss2_2 = L1_loss(gram(style_2), gram(fake_2))/tf.cast(c_2, tf.float32)
    return weight[0] * loss2_2, weight[1] * loss3_3, weight[2] * loss4_4


def Lab_color_loss(photo, fake, weight=1.0):
    photo = (photo + 1.0) / 2.0
    fake = (fake + 1.0) / 2.0
    photo = rgb_to_lab(photo)
    fake = rgb_to_lab(fake)
    # L: 0~100, a: -128~127, b: -128~127
    loss = 2. * L1_loss(photo[:, :, :, 0]/100., fake[:, :, :, 0]/100.) + L1_loss((photo[:, :, :, 1]+128.)/255., (fake[:, :, :, 1]+128.)/255.) \
        + L1_loss((photo[:, :, :, 2]+128.)/255., (fake[:, :, :, 2]+128.)/255.)
    return weight * loss


def total_variation_loss(inputs):
    """
    A smooth loss in fact. Like the smooth prior in MRF.
    V(y) = || y_{n+1} - y_n ||_2
    """
    dh = inputs[:, :-1, ...] - inputs[:, 1:, ...]
    dw = inputs[:, :, :-1, ...] - inputs[:, :, 1:, ...]
    size_dh = tf.size(dh, out_type=tf.float32)
    size_dw = tf.size(dw, out_type=tf.float32)
    return tf.nn.l2_loss(dh) / size_dh + tf.nn.l2_loss(dw) / size_dw


def rgb2yuv(rgb):
    """
    Convert RGB image into YUV https://en.wikipedia.org/wiki/YUV
    """
    rgb = (rgb + 1.0)/2.0
    return tf.image.rgb_to_yuv(rgb)


def yuv_color_loss(photo, fake):
    photo = rgb2yuv(photo)
    fake = rgb2yuv(fake)
    return L1_loss(photo[:, :, :, 0], fake[:, :, :, 0]) + Huber_loss(photo[:, :, :, 1], fake[:, :, :, 1]) + Huber_loss(photo[:, :, :, 2], fake[:, :, :, 2])
