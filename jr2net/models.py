import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from jr2net.layers import EncodeLayer, DecodeLayer, GradientDescent, PriorLayer, FACTORS
from jr2net.layers import ConvBlock
from jr2net.utils import dd_cassi, dd_icassi

CONV_BLOCKS = {
    'baseline': ConvBlock,
}


class JR2net:
    def __init__(self, input_size=(512, 512, 31), features=64, num_stages=5,
                 training=True, factors=FACTORS,  prior_factor=1, conv_model='baseline'):

        N, M, L = input_size
        self.input_size = input_size
        self.unrolled_size = (N, M, 1)
        self.features = features
        self.factors = factors
        self.conv_block = CONV_BLOCKS[conv_model]
        self.unrolled = self.get_recons_net(
            num_stages, training=training, prior_factor=prior_factor)

    def get_recons_net(self, num_stages, prior_factor=1, training=True):

        L = self.input_size[-1]

        # outputs = []
        encoder = EncodeLayer(
            self.features, factors=self.factors, CONV_BLOCK=self.conv_block)
        decoder = DecodeLayer(
            self.features, L, factors=self.factors, CONV_BLOCK=self.conv_block)

        Phi = Lambda(lambda x: dd_cassi(x), name='forward_cassi')
        PhiT = Lambda(lambda x: dd_icassi(x), name='inverse_cassi')

        CA = Input(self.input_size, name='coded_aperture')

        if training:
            inputs = Input(self.input_size, name='spectral_image')
            Af = decoder(encoder(inputs))
            Af = Lambda(lambda x: tf.identity(x), name='repre')(Af)
            input_sensor = Phi([inputs, CA])
            X0 = PhiT([input_sensor, CA])
            

        else:
            inputs = Input(self.unrolled_size, name='measurement')
            X0 = PhiT([inputs, CA])

        f = X0
        alpha = encoder(f)

        u = Lambda(lambda x: tf.scalar_mul(0.0, x))(f)

        for i in range(num_stages):
            z = PriorLayer(factor=prior_factor, name=f"PriorLayer_{i}")(f + u)
            t1 = PhiT([Phi([f, CA]), CA])
            t1 = t1 - X0
            t1 = encoder(t1)
            t2 = encoder(f - z + u)
            alpha = GradientDescent(
                name=f"GradientDescent_{i}")([alpha, t1, t2])
            # alpha = Lambda( lambda x:  tfa.image.gaussian_filter2d(x, 3))(alpha) # gaussian blur filter
            f = decoder(alpha)
            # outputs.append(f)
            # u = AdmmUpdate(name=f"AdmmUpdate_{i}")([u , f , z])
            u = u + f - z

        recons = f
        recons = Lambda(lambda x: tf.identity(x), name='recons')(recons)

        if training:
            model = Model([inputs, CA], [recons, Af], name='JR2net')
        else:
            model = Model([inputs, CA], recons, name='JR2net')

        return model
