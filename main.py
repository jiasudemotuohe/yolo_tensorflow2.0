# -*- coding: utf-8 -*-
# @Time    : 2020-04-18 18:09
# @Author  : speeding_motor

import config
import data_generator
import matplotlib
import yolo
import time
import tensorflow as tf
matplotlib.use("TkAgg")
from matplotlib import pyplot
from tensorflow import keras
from yolo_loss import YoloLoss


def train():
    batch_image_names, batch_boxs = data_generator.get_batch_data()

    yolo_model = yolo.YoloModel()
    yolo_loss = YoloLoss()
    optimizer = keras.optimizers.Adam()

    for epoch in range(config.EPOCHS):
        for i in range(batch_image_names.shape[0]//config.BATCH_SIZE):
            batch_labels = data_generator.generate_label_from_box(batch_boxs[i])
            batch_images = data_generator.read_image_from_names(batch_image_names[i])

            # show_image(batch_images[0:100])
            with tf.GradientTape() as tape:

                batch_prediction = yolo_model(batch_images)

                loss = yolo_loss(batch_labels, batch_prediction)

            grads = tape.gradient(loss, yolo_model.trainable_weights)
            optimizer.apply_gradients(zip(grads, yolo_model.trainable_weights))

            print("epoch ={}, loss={}".format(epoch, loss))


def show_image(images):
    pyplot.figure(figsize=(20, 20))
    for i, image in enumerate(images):
        pyplot.subplot(10, 10, i+1)
        pyplot.imshow(images[i].numpy() * 255)
        pyplot.xticks([])
        pyplot.yticks([])

    pyplot.show()


if __name__ == '__main__':
    train()
    print("train done spend time {}".format(time.clock()))


