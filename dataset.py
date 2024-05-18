import os
import scipy.io as sio
import random
import tensorflow as tf
import numpy as np

from tensorflow.data import AUTOTUNE
from jr2net.utils import coded2DTO3D, dd_cassi
from tensorflow.keras import layers

TRANSMITTANCE = 0.3
SIZE = 96
BANDS = 31

VALIDATION_CODED_APERTURE = f"./codes/H_T={TRANSMITTANCE}.mat"

def get_list_imgs(data_path):
    list_imgs = os.listdir(data_path)
    list_imgs = [os.path.join(data_path, img) for img in list_imgs]
    random.shuffle(list_imgs)
    return list_imgs


def generate_H(coded_size=None, transmittance=TRANSMITTANCE):
    H = tf.random.uniform(coded_size, dtype=tf.float32)
    H = tf.cast(H < transmittance, dtype=tf.float32)*1
    H = coded2DTO3D(H, L=BANDS)
    return H


def csi_mapping(x, coded_size, transmittance=TRANSMITTANCE, training=True):
    batch = x.shape[0]
    coded_size = (batch,) + coded_size
    H = generate_H(coded_size, transmittance)
    if training:
        return (x, H), x
    y = dd_cassi([x, H])
    return (y, H), x

def fine_mapping(x, size=None):
    batch = x.shape[0]

    H =  sio.loadmat('Hreal.mat')['H']
    H = tf.constant(H, dtype=tf.float32)
    H = tf.image.random_crop(H, (2, size, size, 31))
    H = tf.expand_dims(H, 0)
    H = tf.tile(H, [batch, 1, 1, 1, 1])
    # return (x, H), x
    # y = dd_cassi([tf.expand_dims(x, 1), H])
    return (x, H), x
    


class DataGen(tf.data.Dataset):

    def _generator(data_path):  

        list_imgs = get_list_imgs(data_path)

        for img_path in list_imgs:
            # x = sio.loadmat(img_path.decode("utf-8"))['img']
            x = sio.loadmat(img_path)['cube']
            x = x / np.max(x)
            yield x

    def __new__(cls, input_size=(512, 512, 31), data_path="../data/kaist/train"):
        output_signature = tf.TensorSpec(shape=input_size, dtype=tf.float32)

        return tf.data.Dataset.from_generator(
            cls._generator,
            output_signature=output_signature,
            args=(data_path,)
        )


def get_val_csi(data_path, size=256, origin_size=(512, 512, 31)):

    H = sio.loadmat(VALIDATION_CODED_APERTURE)['H'][None, ..., None]
    # H = sio.loadmat('Hreal.mat')['H']
    H = tf.cast(H, dtype=tf.float32)
    # H = coded2DTO3D(H)[None, ...]

    # def map_fun(x): return  (x, H[0]), x

    resize = lambda x: tf.image.resize(x, (size, size))
    # def map_fun(x): return ( dd_cassi([x[None, None, ...] ,  H ])[0], H[0]), x
    def map_fun(x): return ( x, H ), x


    dataset = DataGen(input_size=origin_size,
                      data_path=data_path)

    dataset = (
        dataset
        .map(resize, num_parallel_calls=AUTOTUNE)
        .map(map_fun, num_parallel_calls=AUTOTUNE )
        .batch(1)
        .prefetch(AUTOTUNE)
    )

    return dataset



def get_csi_pipeline(data_path, input_size=(512, 512, 31), patches=True, origin_size=(512, 512, 31),
                     batch_size=32, buffer_size=None, cache_dir='', factor=1, training=True):

    M, N, L = input_size
    coded_size = (M, N + L - 1, 1)

    def map_fun(x): return csi_mapping(x, coded_size, training=training)
    # def map_fun(x): return fine_mapping(x, size=size)
    def replicate(x): return tf.tile(x, [factor, 1, 1, 1])

    dataset = DataGen(input_size=origin_size,
                      data_path=data_path).cache(cache_dir)

    if factor > 1:
        dataset = (
            dataset
            .batch(1)
            .map(replicate, num_parallel_calls=AUTOTUNE)
            .unbatch()
        )

    if patches:
        patches = tf.keras.Sequential([
            layers.experimental.preprocessing.RandomCrop(M, N)
        ])
        dataset = dataset.map(patches, num_parallel_calls=AUTOTUNE)

    if buffer_size:
        dataset = dataset.shuffle(buffer_size)

    dataset = (
        dataset
        .batch(batch_size*factor, drop_remainder=True)
        .map(map_fun, num_parallel_calls=AUTOTUNE)
        .prefetch(AUTOTUNE)
    )

    return dataset


def get_pipeline(data_path, input_size=(512, 512, 31), batch_size=32, buffer_size=3, cache_dir=''):

    dataset = DataGen(input_size=input_size, data_path=data_path)
    def map_fun(x): return (x, x)

    pipeline_data = (
        dataset
        .cache(cache_dir)
        .shuffle(buffer_size)  # cache_dir='' guarda el cache en RAM
        .batch(batch_size, drop_remainder=True)
        .map(map_fun, num_parallel_calls=AUTOTUNE)
        .prefetch(buffer_size)
    )

    return pipeline_data
