import tensorflow as tf
from tools.ops import ConvLayer, LADE_D_Layer, lrelu


class Discriminator(tf.Module):
    """AnimeGANv3 Discriminator network."""

    def __init__(self, sn=True, ch=32, name='discriminator'):
        super().__init__(name=name)

        self.conv_0 = ConvLayer(3, ch, kernel=7, stride=1, sn=sn, name='conv_0')

        self.blocks = []
        in_ch = ch
        for i in range(3):
            out_ch = in_ch  # for stride-2 conv
            block = {
                'conv_s2': ConvLayer(in_ch, out_ch, kernel=3, stride=2, sn=sn, name=f'conv_s2_{i}'),
                'lade_a': LADE_D_Layer(out_ch, sn=sn, name=f'{name}a{i}'),
                'conv_s1': ConvLayer(out_ch, out_ch * 2, kernel=3, stride=1, sn=sn, name=f'conv_s1_{i}'),
                'lade_b': LADE_D_Layer(out_ch * 2, sn=sn, name=f'{name}b{i}'),
            }
            self.blocks.append(block)
            in_ch = out_ch * 2

        self.logit_conv = ConvLayer(in_ch, 1, kernel=1, stride=1, sn=sn, name='D_logit')

    def __call__(self, x):
        x = self.conv_0(x)
        x = lrelu(x)

        for block in self.blocks:
            x = block['conv_s2'](x)
            x = block['lade_a'](x)
            x = lrelu(x)

            x = block['conv_s1'](x)
            x = block['lade_b'](x)
            x = lrelu(x)

        x = self.logit_conv(x)
        return x


def D_net(x_init, sn, ch=32, discriminator_instance=None):
    """Backward-compatible functional wrapper."""
    if discriminator_instance is None:
        discriminator_instance = Discriminator(sn=sn, ch=ch)
    return discriminator_instance(x_init)
