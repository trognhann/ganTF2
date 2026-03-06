import tensorflow as tf
import os
from .tf_color_ops import rgb_to_lab
from .vgg19 import Vgg19


# Xavier : tf.initializers.glorot_uniform()
# He : tf.initializers.variance_scaling()
# Normal : tf.random_normal_initializer(mean=0.0, stddev=0.02)
# l2_decay : tf.keras.regularizers.l2(0.0001)


weight_init = tf.initializers.GlorotUniform()


def l2_regularizer(weight=0.0001):
    def l2(x):
        return weight * tf.nn.l2_loss(x)
    return l2


weight_regularizer_fn = l2_regularizer(0.0001)


##################################################################################
# Activation function
##################################################################################

def lrelu(x, alpha=0.2):
    return tf.nn.leaky_relu(x, alpha)


def relu(x):
    return tf.nn.relu(x)


def tanh(x):
    return tf.math.tanh(x)


def sigmoid(x):
    return tf.math.sigmoid(x)


def h_swish(x):
    return x * tf.nn.relu6(x+3)/6.0


##################################################################################
# Normalization classes (tf.Module)
##################################################################################

class InstanceNorm(tf.Module):
    def __init__(self, num_features, name='instance_norm'):
        super().__init__(name=name)
        self.scale = tf.Variable(tf.ones([num_features]), name='scale')
        self.offset = tf.Variable(tf.zeros([num_features]), name='offset')
        self.epsilon = 1e-5

    def __call__(self, x):
        mean, variance = tf.nn.moments(x, [1, 2], keepdims=True)
        inv = tf.math.rsqrt(variance + self.epsilon)
        normalized = (x - mean) * inv
        return self.scale * normalized + self.offset


class LayerNorm(tf.Module):
    def __init__(self, num_features, name='layer_norm'):
        super().__init__(name=name)
        self.scale = tf.Variable(tf.ones([num_features]), name='scale')
        self.offset = tf.Variable(tf.zeros([num_features]), name='offset')
        self.epsilon = 1e-5

    def __call__(self, x):
        mean, variance = tf.nn.moments(x, [1, 2, 3], keepdims=True)
        inv = tf.math.rsqrt(variance + self.epsilon)
        normalized = (x - mean) * inv
        return self.scale * normalized + self.offset


class BatchNorm(tf.Module):
    def __init__(self, num_features, decay=0.999, epsilon=0.001, name='batch_norm'):
        super().__init__(name=name)
        self.scale = tf.Variable(tf.ones([num_features]), name='scale')
        self.beta = tf.Variable(tf.zeros([num_features]), name='beta')
        self.pop_mean = tf.Variable(tf.zeros([num_features]), trainable=False, name='pop_mean')
        self.pop_var = tf.Variable(tf.ones([num_features]), trainable=False, name='pop_var')
        self.decay = decay
        self.epsilon = epsilon

    def __call__(self, x, is_training=True):
        if is_training:
            batch_mean, batch_var = tf.nn.moments(x, [0, 1, 2])
            self.pop_mean.assign(self.pop_mean * self.decay + batch_mean * (1 - self.decay))
            self.pop_var.assign(self.pop_var * self.decay + batch_var * (1 - self.decay))
            return tf.nn.batch_normalization(x, batch_mean, batch_var, self.beta, self.scale, self.epsilon)
        else:
            return tf.nn.batch_normalization(x, self.pop_mean, self.pop_var, self.beta, self.scale, self.epsilon)


##################################################################################
# Spectral normalization
##################################################################################

class SpectralNorm(tf.Module):
    """Wraps a weight variable with spectral normalization."""
    def __init__(self, w_shape, name='spectral_norm'):
        super().__init__(name=name)
        self.w = tf.Variable(weight_init(w_shape), name='kernel')
        self.u = tf.Variable(
            tf.random.truncated_normal([1, w_shape[-1]]),
            trainable=False, name='u')

    def __call__(self):
        w_shape = self.w.shape.as_list()
        w = tf.reshape(self.w, [-1, w_shape[-1]])

        u_hat = self.u
        v_ = tf.matmul(u_hat, tf.transpose(w))
        v_hat = l2_norm(v_)
        u_ = tf.matmul(v_hat, w)
        u_hat = l2_norm(u_)

        sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))
        w_norm = w / sigma

        self.u.assign(u_hat)
        w_norm = tf.reshape(w_norm, w_shape)
        return w_norm


def l2_norm(v, eps=1e-12):
    return v / (tf.reduce_sum(v ** 2) ** 0.5 + eps)


##################################################################################
# Layer classes (tf.Module)
##################################################################################

class ConvLayer(tf.Module):
    """Conv layer with optional spectral norm, padding, and bias."""
    def __init__(self, in_channels, channels, kernel=4, stride=2,
                 sn=False, pad_type='reflect', use_bias=False, name='conv'):
        super().__init__(name=name)
        self.kernel_size = kernel
        self.stride = stride
        self.pad_type = pad_type
        self.use_bias = use_bias
        self.sn = sn

        # Compute padding
        if (kernel - stride) % 2 == 0:
            pad = (kernel - stride) // 2
            self.pad_top = self.pad_bottom = self.pad_left = self.pad_right = pad
        else:
            pad = (kernel - stride) // 2
            self.pad_bottom = self.pad_right = pad
            self.pad_top = kernel - stride - self.pad_bottom
            self.pad_left = kernel - stride - self.pad_right

        w_shape = [kernel, kernel, in_channels, channels]
        if sn:
            self.sn_wrapper = SpectralNorm(w_shape, name='sn')
        else:
            self.w = tf.Variable(weight_init(w_shape), name='kernel')

        if use_bias:
            self.bias = tf.Variable(tf.zeros([channels]), name='bias')

    def __call__(self, x):
        if self.pad_type == 'zero':
            x = tf.pad(x, [[0, 0], [self.pad_top, self.pad_bottom],
                       [self.pad_left, self.pad_right], [0, 0]])
        elif self.pad_type == 'reflect':
            x = tf.pad(x, [[0, 0], [self.pad_top, self.pad_bottom],
                       [self.pad_left, self.pad_right], [0, 0]], mode='REFLECT')

        w = self.sn_wrapper() if self.sn else self.w
        x = tf.nn.conv2d(input=x, filters=w, strides=[1, self.stride, self.stride, 1], padding='VALID')

        if self.use_bias:
            x = tf.nn.bias_add(x, self.bias)
        return x


class Conv2DLayer(tf.Module):
    """Conv2D layer with reflect padding."""
    def __init__(self, in_channels, filters, kernel_size=3, strides=1,
                 use_bias=False, activation_fn=None, name='conv2d'):
        super().__init__(name=name)
        self.kernel_size = kernel_size
        self.strides = strides
        self.activation_fn = activation_fn
        self.use_bias = use_bias

        if (kernel_size - strides) % 2 == 0:
            pad = (kernel_size - strides) // 2
            self.pad_top = self.pad_bottom = self.pad_left = self.pad_right = pad
        else:
            pad = (kernel_size - strides) // 2
            self.pad_bottom = self.pad_right = pad
            self.pad_top = kernel_size - strides - self.pad_bottom
            self.pad_left = kernel_size - strides - self.pad_right

        self.w = tf.Variable(
            weight_init([kernel_size, kernel_size, in_channels, filters]),
            name='kernel')

        if use_bias:
            self.bias_var = tf.Variable(tf.zeros([filters]), name='bias')

    def __call__(self, x):
        x = tf.pad(x, [[0, 0], [self.pad_top, self.pad_bottom],
                   [self.pad_left, self.pad_right], [0, 0]], mode="REFLECT")
        x = tf.nn.conv2d(input=x, filters=self.w,
                         strides=[1, self.strides, self.strides, 1], padding='VALID')
        if self.use_bias:
            x = tf.nn.bias_add(x, self.bias_var)
        if self.activation_fn:
            x = self.activation_fn(x)
        return x


class LADE_Layer(tf.Module):
    """Learnable Affine Denormalization layer."""
    def __init__(self, in_channels, name='LADE'):
        super().__init__(name=name)
        self.conv = Conv2DLayer(in_channels, in_channels, 1, 1, name='conv_IN')
        self.eps = 1e-5

    def __call__(self, x):
        tx = self.conv(x)
        t_mean, t_sigma = tf.nn.moments(tx, axes=[1, 2], keepdims=True)
        in_mean, in_sigma = tf.nn.moments(x, axes=[1, 2], keepdims=True)
        x_in = (x - in_mean) / (tf.sqrt(in_sigma + self.eps))
        x = x_in * (tf.sqrt(t_sigma + self.eps)) + t_mean
        return x


class LADE_D_Layer(tf.Module):
    """LADE for Discriminator (with optional spectral norm)."""
    def __init__(self, in_channels, sn=False, name='LADE_D'):
        super().__init__(name=name)
        self.conv = ConvLayer(in_channels, in_channels, kernel=1, stride=1,
                              sn=sn, pad_type='zero', name='conv_IN')
        self.eps = 1e-5

    def __call__(self, x):
        tx = self.conv(x)
        t_mean, t_sigma = tf.nn.moments(tx, axes=[1, 2], keepdims=True)
        in_mean, in_sigma = tf.nn.moments(x, axes=[1, 2], keepdims=True)
        x_in = (x - in_mean) / (tf.sqrt(in_sigma + self.eps))
        x = x_in * (tf.sqrt(t_sigma + self.eps)) + t_mean
        return x


class ConvLADELrelu(tf.Module):
    """Conv + LADE + LeakyReLU block."""
    def __init__(self, in_channels, filters, kernel_size=3, strides=1, name='conv_LADE'):
        super().__init__(name=name)
        self.conv = Conv2DLayer(in_channels, filters, kernel_size, strides, name='conv')
        self.lade = LADE_Layer(filters, name='LADE')

    def __call__(self, x):
        x = self.conv(x)
        x = self.lade(x)
        return lrelu(x)


class ExternalAttentionV3(tf.Module):
    """External Attention module."""
    def __init__(self, in_channels, k=128, name='External_attention'):
        super().__init__(name=name)
        c = in_channels
        self.conv1 = Conv2DLayer(c, c, 1, 1, name='conv1')
        self.w_kernel = tf.Variable(
            weight_init([1, c, k]),
            name='kernel')
        self.conv2 = Conv2DLayer(c, c, 1, 1, name='conv2')
        self.bn = BatchNorm(c, name='bn')
        self.c = c

    def __call__(self, x, is_training):
        idn = x
        b = tf.shape(x)[0]
        h = tf.shape(x)[1]
        w = tf.shape(x)[2]
        c = self.c

        x = self.conv1(x)
        x = tf.reshape(x, shape=[b, -1, c])
        attn = tf.nn.conv1d(x, self.w_kernel, stride=1, padding='VALID')
        attn = tf.nn.softmax(attn, axis=1)
        attn = attn / (1e-9 + tf.reduce_sum(attn, axis=2, keepdims=True))

        w_kernel_t = tf.transpose(self.w_kernel, perm=[0, 2, 1])
        x = tf.nn.conv1d(attn, w_kernel_t, stride=1, padding='VALID')
        x = tf.reshape(x, [b, h, w, c])
        x = self.conv2(x)
        x = self.bn(x, is_training)
        x = x + idn
        out = lrelu(x)
        return out


##################################################################################
# Sampling
##################################################################################

def flatten(x):
    return tf.reshape(x, [tf.shape(x)[0], -1])


def global_avg_pooling(x, keepdims=True):
    gap = tf.reduce_mean(x, axis=[1, 2], keepdims=keepdims)
    return gap


def global_max_pooling(x, keepdims=True):
    gmp = tf.reduce_max(x, axis=[1, 2], keepdims=keepdims)
    return gmp


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
    return tf.reduce_mean(tf.keras.losses.huber(x, y, delta=delta))


def generator_loss(fake):
    fake_loss = tf.reduce_mean(tf.square(fake - 0.9))
    return fake_loss


def discriminator_loss(anime_logit, fake_logit):
    # lsgan :
    anime_gray_logit_loss = tf.reduce_mean(tf.square(anime_logit - 0.9))
    fake_gray_logit_loss = tf.reduce_mean(tf.square(fake_logit-0.1))
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
    c = x.shape[-1]
    return L1_loss(x, y)/tf.cast(c, tf.float32)


def con_loss(real, fake, weight=1.0):
    return weight * VGG_LOSS(real, fake)


def region_smoothing_loss(seg, fake, weight):
    return VGG_LOSS(seg, fake) * weight


def style_loss(style, fake, weight):
    style_feat = vgg19.build(style)
    fake_feat = vgg19.build(fake)
    return weight * L1_loss(gram(style_feat), gram(fake_feat))/tf.cast(style_feat.shape[-1], tf.float32)


def style_loss_decentralization_3(style, fake, weight):
    style_4, style_3, style_2 = vgg19.build_multi(style)
    fake_4, fake_3, fake_2 = vgg19.build_multi(fake)
    dim = [1, 2]
    style_2 -= tf.reduce_mean(style_2, axis=dim, keepdims=True)
    fake_2 -= tf.reduce_mean(fake_2, axis=dim, keepdims=True)
    c_2 = fake_2.shape[-1]

    style_3 -= tf.reduce_mean(style_3, axis=dim, keepdims=True)
    fake_3 -= tf.reduce_mean(fake_3, axis=dim, keepdims=True)
    c_3 = fake_3.shape[-1]

    style_4 -= tf.reduce_mean(style_4, axis=dim, keepdims=True)
    fake_4 -= tf.reduce_mean(fake_4, axis=dim, keepdims=True)
    c_4 = fake_4.shape[-1]

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
    size_dh = tf.cast(tf.size(dh), tf.float32)
    size_dw = tf.cast(tf.size(dw), tf.float32)
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
