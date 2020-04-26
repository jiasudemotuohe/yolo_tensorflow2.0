# -*- coding: utf-8 -*-
# @Time    : 2020-04-19 15:08
# @Author  : speeding_motor

from tensorflow import keras
from config import GRID_SIZE
import tensorflow as tf


class YoloModel(keras.Model):

    def __init__(self):
        super(YoloModel, self).__init__()
        alpha = 0.1

        self.model = keras.Sequential()

        self.model.add(keras.layers.Conv2D(filters=64, kernel_size=(7, 7), strides=2, padding='same', input_shape=[448, 448, 3]))
        self.model.add(keras.layers.LeakyReLU(alpha))
        self.model.add(self.__maxpooling_layer(pool_sizes=(2, 2), strides=2))

        self.model.add(self.__convolution__layer(filters=192, kernel_size=(3, 3), strides=1, padding='same'))
        self.model.add(keras.layers.LeakyReLU(alpha))
        self.model.add(self.__maxpooling_layer(pool_sizes=(2, 2), strides=2))

        self.model.add(self.__convolution__layer(filters=128, kernel_size=(1, 1), strides=1))
        self.model.add(keras.layers.LeakyReLU(alpha))
        self.model.add(self.__convolution__layer(filters=256, kernel_size=(3, 3), strides=1, padding="same"))
        self.model.add(keras.layers.LeakyReLU(alpha))
        self.model.add(self.__convolution__layer(filters=256, kernel_size=(1, 1), strides=1, padding="same"))
        self.model.add(keras.layers.LeakyReLU(alpha))
        self.model.add(self.__convolution__layer(filters=512, kernel_size=(3, 3), strides=1, padding="same"))
        self.model.add(keras.layers.LeakyReLU(alpha))
        self.model.add(self.__maxpooling_layer(pool_sizes=(2, 2), strides=2))

        self.model.add(self.__convolution__layer(filters=256, kernel_size=(1, 1), strides=1, padding='same'))
        self.model.add(keras.layers.LeakyReLU(alpha))
        self.model.add(self.__convolution__layer(filters=512, kernel_size=(3, 3), strides=1, padding='same'))
        self.model.add(keras.layers.LeakyReLU(alpha))
        self.model.add(self.__convolution__layer(filters=256, kernel_size=(1, 1), strides=1, padding='same'))
        self.model.add(keras.layers.LeakyReLU(alpha))
        self.model.add(self.__convolution__layer(filters=512, kernel_size=(3, 3), strides=1, padding='same'))
        self.model.add(keras.layers.LeakyReLU(alpha))
        self.model.add(self.__convolution__layer(filters=256, kernel_size=(1, 1), strides=1, padding='same'))
        self.model.add(keras.layers.LeakyReLU(alpha))
        self.model.add(self.__convolution__layer(filters=512, kernel_size=(3, 3), strides=1, padding='same'))
        self.model.add(keras.layers.LeakyReLU(alpha))
        self.model.add(self.__convolution__layer(filters=256, kernel_size=(1, 1), strides=1, padding='same'))
        self.model.add(keras.layers.LeakyReLU(alpha))
        self.model.add(self.__convolution__layer(filters=512, kernel_size=(3, 3), strides=1, padding='same'))
        self.model.add(keras.layers.LeakyReLU(alpha))

        self.model.add(self.__convolution__layer(filters=512, kernel_size=(1, 1), strides=1, padding='same'))
        self.model.add(keras.layers.LeakyReLU(alpha))
        self.model.add(self.__convolution__layer(filters=1024, kernel_size=(3, 3), strides=1, padding='same'))
        self.model.add(keras.layers.LeakyReLU(alpha))
        self.model.add(self.__maxpooling_layer(pool_sizes=(2, 2), strides=2))

        self.model.add(self.__convolution__layer(filters=512, kernel_size=(1, 1), strides=1, padding='same'))
        self.model.add(keras.layers.LeakyReLU(alpha))
        self.model.add(self.__convolution__layer(filters=1024, kernel_size=(3, 3), strides=1, padding='same'))
        self.model.add(keras.layers.LeakyReLU(alpha))
        self.model.add(self.__convolution__layer(filters=512, kernel_size=(1, 1), strides=1, padding='same'))
        self.model.add(keras.layers.LeakyReLU(alpha))
        self.model.add(self.__convolution__layer(filters=1024, kernel_size=(3, 3), strides=1, padding='same'))
        self.model.add(keras.layers.LeakyReLU(alpha))
        self.model.add(self.__convolution__layer(filters=1024, kernel_size=(3, 3), strides=1, padding='same'))
        self.model.add(keras.layers.ReLU(alpha))
        self.model.add(self.__convolution__layer(filters=1024, kernel_size=(3, 3), strides=(2, 2), padding='same'))

        self.model.add(keras.layers.Flatten())
        self.model.add(keras.layers.Dense(4096, kernel_initializer='glorot_uniform'))
        self.model.add(keras.layers.LeakyReLU(alpha))

        self.model.add(keras.layers.Dense(1225, trainable=True, kernel_initializer='glorot_uniform'))
        self.model.add(keras.layers.ReLU())

        self.model.add(keras.layers.Reshape([GRID_SIZE, GRID_SIZE, 25]))

        # self.build()

    def call(self, inputs):
        return self.model(inputs)

    def __convolution__layer(self, filters, kernel_size, strides=1, padding="valid"):
        layer = keras.layers.Conv2D(filters=filters, kernel_size=kernel_size,
                                    strides=strides, padding=padding)
        return layer

    def __maxpooling_layer(self, pool_sizes, strides=1):
        max_pool = keras.layers.MaxPool2D(pool_sizes, strides=strides)
        return max_pool