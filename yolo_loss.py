# -*- coding: utf-8 -*-
# @Time    : 2020-04-21 14:09
# @Author  : speeding_motor

from tensorflow import keras
import tensorflow as tf
import config
import numpy as np


class YoloLoss(keras.losses.Loss):
    def __init__(self):
        super().__init__()
        self.classify_scale = 5.0

    def call(self, y_true, y_pred):
        """
        return loss: tensor with shape [1, batch_size]
        y_true: [batch_size, grid_size, grid_size, 25]

        """
        y_true = tf.dtypes.cast(y_true, dtype=tf.double)
        y_pred = tf.dtypes.cast(y_pred, dtype=tf.double)

        coordinate_loss = self.__localization__loss(y_true, y_pred)
        classes_loss = self.__classsification_loss(y_true, y_pred)
        object_loss = self.__object_probability_loss(y_true, y_pred)

        return coordinate_loss + classes_loss + object_loss

    def __object_probability_loss(self, y_true, y_pred):
        y_ture_object_proba = tf.gather(params=y_true, indices=[0], axis=3)
        y_pred_object_proba = tf.gather(params=y_pred, indices=[0], axis=3)

        loss = tf.square(y_ture_object_proba - y_pred_object_proba) * self.classify_scale
        return tf.reduce_sum(input_tensor=loss, axis=[1, 2, 3], keepdims=False, name="object_probability_loss")

    def __classsification_loss(self, y_true, y_pred):
        """
        return: the classify loss with the classify

        """
        classs_index = np.array(range(len(config.PASCAL_VOC_CLASSES)))+5
        true_class = tf.gather(params=y_true, indices=classs_index, axis=3)
        pred_class = tf.gather(params=y_true, indices=classs_index, axis=3)

        loss = tf.square(true_class - pred_class)
        return tf.reduce_sum(input_tensor=loss, axis=[1, 2, 3], keepdims=False, name="classification_loss")

    def __localization__loss(self, y_true, y_pred):
        """
        return: the localize loss with the box, loss shape [1, batch_size]
        localization_loss = (x - x_pred)^2 + (y - y_pred)^2 + (w - w_pred)^2 + (h - h_pred)^2

        """
        true_xy = tf.gather(params=y_true, indices=[1, 2], axis=3)
        pred_xy = tf.gather(params=y_pred, indices=[1, 2], axis=3)

        true_wh = tf.gather(params=y_true, indices=[3, 4], axis=3)
        pred_wh = tf.gather(params=y_pred, indices=[3, 4], axis=3)

        # shape: [batch_size, grid_size, grid_size, 2]
        loss = tf.square((true_xy - pred_xy)) + tf.square(tf.sqrt(true_wh) - tf.sqrt(pred_wh))

        return tf.reduce_sum(input_tensor=loss, axis=[1, 2, 3], keepdims=False, name="box_coordinate_loss")



