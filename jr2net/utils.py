import gc
import math
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import Callback
from tensorflow.keras import backend as k

TV_KERNEL = np.zeros((3,3,1,1))

TV_KERNEL[:,:,0,0] = np.array(
    [
    [ 0,-1, 0],
    [-1, 2, 0],    
    [ 0, 0, 0], 
    ]
)


def sof_tresh(V, tau):    
    V1 = (V > tau)*(V - tau)
    V2 = (V < -tau)*(V + tau)
    resul = V1 + V2
    return resul

@tf.function
def tv_prior(inputs):
    dy, dx = tf.image.image_gradients(inputs)
    tv = tf.add(tf.abs(dy), tf.abs(dx))
    return tv

@tf.function
def dd_cassi(x):
    inputs, H = x
    y = tf.multiply(H, inputs)
    y = tf.reduce_sum(y, -1, keepdims=True)
    return y

@tf.function    
def dd_icassi(x):
    y, H = x

    Hn = tf.divide(H, tf.add(tf.reduce_sum(H, -1, keepdims=True), 1e-12))
    y = tf.multiply(Hn, y) 
    return y

def ChannelwiseConv2D(inputs, kernel):
    inputs = tf.split(inputs , [1]*inputs.shape[-1] , axis=-1)
    output = [tf.nn.conv2d(i, kernel, strides=[1,1,1,1], padding="SAME") for i in inputs] 
    output = tf.concat(output,-1)
    return output


def ImgGrad(inputs):
    kernel = TV_KERNEL
    output = ChannelwiseConv2D(inputs, kernel)
    return output 

def ImgGradT(inputs):
    kernel = TV_KERNEL[::-1,::-1,:,:]
    output = ChannelwiseConv2D(inputs, kernel)
    return output 

@tf.function
def coded2DTO3D(CA, L=31):
    _ , N, M, _ = CA.shape
    H = tf.concat([CA[:, :, i:N+i, :] for i in range(L)], -1)
    return H


@tf.function
def spatial_spectral_loss(loss_weights=[1, 1]):

    t1 , t2 = loss_weights/np.sum(loss_weights)

    def loss(y_true, y_pred):

        spatial_loss =tf.reduce_mean(tf.norm(y_pred - y_true, ord=1)) + tf.reduce_mean(1-tf.image.ssim_multiscale(y_pred,y_true,1))

        a_b = tf.math.reduce_sum(tf.multiply(y_pred,y_true),axis=-1)


        mag_a = tf.sqrt(tf.reduce_sum(y_pred**2,axis=-1))
        mag_b = tf.sqrt(tf.reduce_sum(y_true**2,axis=-1))


        spectral_loss = tf.reduce_mean(tf.abs(a_b-tf.multiply(mag_a,mag_b)))     

        val = t1*spectral_loss + t2*spatial_loss
        return val

    return loss


def SAM(y_true,y_pred):

    pp = tf.sqrt(tf.reduce_sum(tf.pow(y_true,2), 2, keepdims=True))+ 2.2e-16
    pp2 = tf.sqrt(tf.reduce_sum(tf.pow(y_pred,2), 2, keepdims=True)) + 2.2e-16
    y_true = tf.divide(y_true,pp)
    y_pred = tf.divide(y_pred,pp2)
    z = tf.reduce_sum(tf.multiply(y_true,y_pred),2)
    z = tf.reduce_mean(tf.acos(z-2.2e-16)*180/ math.pi)

    return z


def addGaussianNoise(y,SNR):
    sigma = np.sum(np.power(y,2))/(np.product(y.shape)*10**(SNR/10))
    w = np.random.normal(0, np.sqrt(sigma),size =y.shape)
    return y+w


class ClearMemory(Callback):
    def on_epoch_end(self, epoch, logs=None):
        gc.collect()
        k.clear_session()