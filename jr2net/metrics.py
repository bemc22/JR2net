import math
import tensorflow as tf
from jr2net.utils import tv_prior
import numpy as np

def prior_loss(y_true, y_pred):
    y_true = tv_prior(y_true)
    return tf.keras.metrics.mean_squared_error(y_true, y_pred)

def psnr(y_true, y_pred):
    return tf.reduce_mean(tf.image.psnr(y_true, y_pred, max_val=1))

def ssim(y_true, y_pred):
    return tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1))

def SAM(y_true, y_pred):

    pp = tf.sqrt(tf.reduce_sum(tf.pow(y_true, 2), -1, keepdims=True)) 
    pp2 = tf.sqrt(tf.reduce_sum(tf.pow(y_pred, 2), -1, keepdims=True)) 

    num = tf.reduce_sum(tf.multiply(y_true, y_pred), -1, keepdims=True)
    denom = tf.multiply( pp, pp2) + 1e-12
    z = tf.divide(num, denom )
    z = tf.maximum( -1., tf.minimum( 1., z) )
    z = tf.acos(z)
    return tf.reduce_mean(z)


# NUMPY VERSION SAM METRIC
def SAM_numpy(y_true, y_pred):

    pp = np.sqrt(np.sum(np.power(y_true, 2), -1, keepdims=True)) 
    pp2 = np.sqrt(np.sum(np.power(y_pred, 2), -1, keepdims=True)) 

    num = np.sum(np.multiply(y_true, y_pred), -1, keepdims=True)
    denom = np.multiply( pp, pp2)
    z = np.divide(num, denom )
    z = np.maximum( -1., np.minimum( 1., z) )
    z = np.mean(np.arccos(z))

    return z

    