# -*- coding: utf-8 -*-
# @Time    : 2020-04-18 23:02
# @Author  : speeding_motor

import tensorflow as tf
import numpy as np
import config


def get_train_test_data():
    """
    this function slove the following question

    first: generate the bath data from the pascal.txt file
    second: parse the batch lines txt to image data、class labels、box

    """
    text_data = tf.data.TextLineDataset(config.PASCAL_TXT_FILE)

    batch_lines_data = text_data.batch(batch_size=config.BATCH_SIZE, drop_remainder=False)
    batch_images = []
    batch_labels = []
    for batch_data in batch_lines_data:
        images, labels = parse_batch_data(batch_data)
        batch_images.append(images)
        batch_labels.append(labels)

    return np.asarray(batch_images), batch_labels


def __parse_singal_line(line):
    # here we need to add the classes and boxs, a picture maybe have mutilple box

    name = line[0]
    classes = line[1:]

    label = np.zeros(len(config.PASCAL_VOC_CLASSES) * 5)
    for i in range(int(len(classes) / 5)):
        index = i * 5

        class_id = int(classes[index])
        xmin = int(float(classes[index + 1]))
        ymin = int(float(classes[index + 2]))
        xmax = int(float(classes[index + 3]))
        ymax = int(float(classes[index + 4]))

        class_index = (class_id - 1) * 5
        label[class_index] = 1
        label[class_index + 1] = xmin
        label[class_index + 2] = ymin
        label[class_index + 3] = xmax
        label[class_index + 4] = ymax

    return name, label


def parse_batch_data(batch_data):
    image_names = []
    labels = []
    for line in batch_data:
        line = line.numpy().decode(encoding="utf-8")
        line = line.strip().split(" ")

        name, label = __parse_singal_line(line)
        image_names.append(name)
        labels.append(label)

    return image_names, labels


if __name__ == '__main__':
    get_train_test_data()