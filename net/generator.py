import tensorflow as tf
from tools.ops import ConvLADELrelu, ExternalAttentionV3, Conv2DLayer


class Generator(tf.Module):
    """AnimeGANv3 Generator network with base, support, and main branches."""

    def __init__(self, name='generator'):
        super().__init__(name=name)

        # Base encoder
        self.base_conv0 = ConvLADELrelu(3, 32, 7, name='base_conv0')         # 256
        self.base_conv1_s2 = ConvLADELrelu(32, 32, strides=2, name='base_conv1_s2')  # 128
        self.base_conv1_s1 = ConvLADELrelu(32, 64, name='base_conv1_s1')
        self.base_conv2_s2 = ConvLADELrelu(64, 64, strides=2, name='base_conv2_s2')  # 64
        self.base_conv2_s1 = ConvLADELrelu(64, 128, name='base_conv2_s1')
        self.base_conv3_s2 = ConvLADELrelu(128, 128, strides=2, name='base_conv3_s2')  # 32
        self.base_conv3_s1 = ConvLADELrelu(128, 128, name='base_conv3_s1')

        # Support decoder  (skip connections use addition, matching channels)
        self.s_ext_attn = ExternalAttentionV3(128, name='support_ext_attn')
        self.s_conv4_1 = ConvLADELrelu(128, 128, name='support_conv4_1')
        self.s_conv4_2 = ConvLADELrelu(128, 128, name='support_conv4_2')   # +x2 (128ch)
        self.s_conv5_1 = ConvLADELrelu(128, 64, name='support_conv5_1')
        self.s_conv5_2 = ConvLADELrelu(64, 64, name='support_conv5_2')     # +x1 (64ch)
        self.s_conv6_1 = ConvLADELrelu(64, 32, name='support_conv6_1')
        self.s_conv6_2 = ConvLADELrelu(32, 32, name='support_conv6_2')     # +x0 (32ch)
        self.s_conv_final = Conv2DLayer(32, 3, 7, 1, name='support_conv_final')

        # Main decoder  (skip connections use addition, matching channels)
        self.m_ext_attn = ExternalAttentionV3(128, name='main_ext_attn')
        self.m_conv4_1 = ConvLADELrelu(128, 128, name='main_conv4_1')
        self.m_conv4_2 = ConvLADELrelu(128, 128, name='main_conv4_2')
        self.m_conv5_1 = ConvLADELrelu(128, 64, name='main_conv5_1')
        self.m_conv5_2 = ConvLADELrelu(64, 64, name='main_conv5_2')
        self.m_conv6_1 = ConvLADELrelu(64, 32, name='main_conv6_1')
        self.m_conv6_2 = ConvLADELrelu(32, 32, name='main_conv6_2')
        self.m_conv_final = Conv2DLayer(32, 3, 7, 1, name='main_conv_final')

    def __call__(self, inputs, is_training):
        # Base encoder
        x0 = self.base_conv0(inputs)         # 256, 32ch
        x1 = self.base_conv1_s2(x0)          # 128, 32ch
        x1 = self.base_conv1_s1(x1)          # 128, 64ch
        x2 = self.base_conv2_s2(x1)          # 64, 64ch
        x2 = self.base_conv2_s1(x2)          # 64, 128ch
        x3 = self.base_conv3_s2(x2)          # 32, 128ch
        x3 = self.base_conv3_s1(x3)          # 32, 128ch

        # Support branch
        s_x3 = self.s_ext_attn(x3, is_training)
        s_x4 = tf.image.resize(s_x3, [2 * tf.shape(s_x3)[1], 2 * tf.shape(s_x3)[2]])  # 64
        s_x4 = self.s_conv4_1(s_x4)
        s_x4 = self.s_conv4_2(s_x4 + x2)   # addition skip (both 128ch)

        s_x5 = tf.image.resize(s_x4, [2 * tf.shape(s_x4)[1], 2 * tf.shape(s_x4)[2]])  # 128
        s_x5 = self.s_conv5_1(s_x5)
        s_x5 = self.s_conv5_2(s_x5 + x1)   # addition skip (both 64ch)

        s_x6 = tf.image.resize(s_x5, [2 * tf.shape(s_x5)[1], 2 * tf.shape(s_x5)[2]])  # 256
        s_x6 = self.s_conv6_1(s_x6)
        s_x6 = self.s_conv6_2(s_x6 + x0)   # addition skip (both 32ch)

        s_final = self.s_conv_final(s_x6)
        fake_s = tf.math.tanh(s_final)

        # Main branch
        m_x3 = self.m_ext_attn(x3, is_training)
        m_x4 = tf.image.resize(m_x3, [2 * tf.shape(m_x3)[1], 2 * tf.shape(m_x3)[2]])  # 64
        m_x4 = self.m_conv4_1(m_x4)
        m_x4 = self.m_conv4_2(m_x4 + x2)   # addition skip (both 128ch)

        m_x5 = tf.image.resize(m_x4, [2 * tf.shape(m_x4)[1], 2 * tf.shape(m_x4)[2]])  # 128
        m_x5 = self.m_conv5_1(m_x5)
        m_x5 = self.m_conv5_2(m_x5 + x1)   # addition skip (both 64ch)

        m_x6 = tf.image.resize(m_x5, [2 * tf.shape(m_x5)[1], 2 * tf.shape(m_x5)[2]])  # 256
        m_x6 = self.m_conv6_1(m_x6)
        m_x6 = self.m_conv6_2(m_x6 + x0)   # addition skip (both 32ch)

        m_final = self.m_conv_final(m_x6)
        fake_m = tf.math.tanh(m_final)

        return fake_s, fake_m
