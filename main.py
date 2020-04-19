# -*- coding: utf-8 -*-
# @Time    : 2020-04-18 18:09
# @Author  : speeding_motor

import config
import generate_data
import matplotlib
matplotlib.use("TkAgg")
from matplotlib import pyplot


def train():
    batch_image_names, batch_labels = generate_data.get_batch_data()

    for epoch in range(config.EPOCHS):

        for i in range(batch_image_names.shape[0]):
            batch_images = generate_data.read_image_from_names(batch_image_names[i])
            show_image(batch_images[0:9])

            break
        break


def show_image(images):

    for i, image in enumerate(images):
        pyplot.subplot(3, 3, i+1)
        pyplot.imshow(images[i].numpy() * 255)
        pyplot.xlim()
        pyplot.xticks([])
        pyplot.yticks([])

        pyplot.xlabel(None)
        pyplot.ylabel(None)

    pyplot.show()



if __name__ == '__main__':

   train()
   print("running speed time i")


