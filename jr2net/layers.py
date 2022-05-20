import tensorflow as tf
from tensorflow.keras.layers import Conv2D, Activation, Add
from tensorflow.keras.constraints import NonNeg
from tensorflow.python.keras import constraints

KERNEL_REGUL = tf.keras.regularizers.L2(1e-8)
CONV_PARAMS = {
    'padding': 'same',
    'kernel_initializer': 'glorot_uniform',
    'kernel_regularizer': KERNEL_REGUL,
}


FACTORS = [1, 1, 1/2, 1/2, 1/4, 1/8]

class ConvBlock(tf.keras.layers.Layer):
    def __init__(self, feature, activation='relu'):
        super(ConvBlock, self).__init__()

        self.conv = Conv2D(feature, 3, **CONV_PARAMS)
        self.activation = Activation(activation)

    def call(self, inputs):

        x = self.conv(inputs)
        x = self.activation(x)
        return x

class EncodeLayer(tf.keras.layers.Layer):
    def __init__(self, feature, name='Encoder', factors=FACTORS, CONV_BLOCK=ConvBlock, **kwargs):
        super(EncodeLayer, self).__init__(name=name, **kwargs)

        self.convs = []

        for factor in factors[:-1]:
            ifeature = int(factor*feature)
            conv = CONV_BLOCK(ifeature)
            self.convs.append(conv)

        self.out_features = int(feature*factors[-1])
        encode = CONV_BLOCK(self.out_features, activation=None)
        self.convs.append(encode)

    def call(self, inputs):
        x = inputs
        for conv in self.convs:
            x = conv(x)
        return x


class DecodeLayer(tf.keras.layers.Layer):
    def __init__(self, feature, L, name='Decoder', CONV_BLOCK=ConvBlock, factors=FACTORS, **kwargs):
        super(DecodeLayer, self).__init__(name=name, **kwargs)

        self.convs = []
        self.L = L

        for factor in factors[-2::-1]:
            ifeature = int(factor*feature)
            conv = CONV_BLOCK(ifeature)
            self.convs.append(conv)

        decode = CONV_BLOCK(L, activation='relu')
        self.convs.append(decode)

    def call(self, inputs):
        x = inputs
        for conv in self.convs:
            x = conv(x)
        return x


class PriorLayer(tf.keras.layers.Layer):
    def __init__(self, factor=2, name='PriorLayer'):
        super(PriorLayer, self).__init__(name=name)

        self.factor = factor

        self.add = Add()

    def build(self, input_shape):
        L = input_shape[-1]
        feature = int(self.factor*L)
        self.conv1 = Conv2D(feature, 3, activation='relu', **CONV_PARAMS)
        self.conv2 = Conv2D(L, 3, activation=None, **CONV_PARAMS)
        self.conv3 = Conv2D(L, 1, activation=None, **CONV_PARAMS)

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.add([inputs, x])
        x = self.conv3(x)
        return x


class GradientDescent(tf.keras.layers.Layer):
    def __init__(self, name='GradientDescent'):
        super(GradientDescent, self).__init__(name=name)

        self.epsilon = self.add_weight(
            name='epsilon', initializer=tf.keras.initializers.Constant(1e-3), trainable=True)

        self.rho = self.add_weight(
            name='rho',   initializer=tf.keras.initializers.Constant(0.1), trainable=True)

    def call(self, inputs):
        a, t1, t2 = inputs
        x = tf.add(t1, tf.scalar_mul(self.rho, t2))
        x = tf.subtract(a, tf.scalar_mul(self.epsilon, x))
        return x


def zero_one_constraint(x): return tf.clip_by_value(x, 0, 1)


class AdamDescent(tf.keras.layers.Layer):
    def __init__(self, num_stage, name='GradientDescent'):
        super(AdamDescent, self).__init__(name=name)

        self.epsilon = self.add_weight(
            name='epsilon', initializer=tf.keras.initializers.Constant(1e-3), trainable=True)

        self.rho = self.add_weight(
            name='rho',   initializer=tf.keras.initializers.Constant(0.1), trainable=True)

        self.beta1 = self.add_weight(
            name='beta1',   initializer=tf.keras.initializers.Constant(0.9),
            trainable=True, constraint=zero_one_constraint)

        self.beta2 = self.add_weight(
            name='beta2',   initializer=tf.keras.initializers.Constant(0.9),
            trainable=True, constraint=zero_one_constraint)

        
    def call(self, inputs):
        a, v, s, t1, t2 = inputs
        Lalpha = tf.add(t1, tf.scalar_mul(self.rho, t2))

        v = tf.scalar_mul( self.beta1, v ) + tf.scalar_mul( 1 - self.beta1, Lalpha)
        s = tf.scalar_mul( self.beta2, s ) + tf.scalar_mul( 1 - self.beta2, tf.square(Lalpha))
        pk  = v / tf.sqrt( s + 1e-6 )
        x = tf.subtract(a, tf.scalar_mul(self.epsilon, pk))

        return x, v, s


class AdmmUpdate(tf.keras.layers.Layer):
    def __init__(self, name='AdmmUpdate'):
        super(AdmmUpdate, self).__init__(name=name)

        self.gamma = self.add_weight(
            name='gamma', initializer=tf.keras.initializers.Constant(1),
            trainable=True, constraint=NonNeg())

    def call(self, inputs):
        u, x, z = inputs
        output = tf.add(u, tf.scalar_mul(self.gamma, tf.subtract(x, z)))
        return output
