# -*- coding: utf-8 -*-
# @Time    : 2020-04-18 23:02
# @Author  : speeding_motor

import numpy as np
import config
import tensorflow as tf


def get_batch_data():
    """
    this function slove the following question

    first: generate the bath data from the data.txt file
    second: parse the batch lines txt to image data、class labels、box

    """
    text_data = tf.data.TextLineDataset(config.PASCAL_TXT_FILE)

    batch_lines_data = text_data.batch(batch_size=config.BATCH_SIZE, drop_remainder=False)
    batch_image_names = []
    batch_boxs = []
    for batch_data in batch_lines_data:
        batch_name, batch_box = parse_batch_data(batch_data)

        batch_image_names.append(batch_name)
        batch_boxs.append(batch_box)

    return np.asarray(batch_image_names), np.asarray(batch_boxs)


def __parse_singal_line(line):
    """
    here we need to add the classes and boxs, a picture maybe have mutilple box, and then we need all the box
    to define the label of each cell of grid on picture

    """
    name = line[0]
    boxs = np.zeros(shape=(config.MAX_NUM_BOXS_PER_IMAGE, 5))

    classes = [int(float(i)) for i in line[1:]]
    classes = np.reshape(classes, newshape=(-1, 5))[0:config.MAX_NUM_BOXS_PER_IMAGE]

    boxs[0:len(classes)] = classes

    # for i in range(int(len(classes) / 5)):
    #     index = i * 5
    #
    #     class_id = int(classes[index])
    #     xmin = int(float(classes[index + 1]))
    #     ymin = int(float(classes[index + 2]))
    #     xmax = int(float(classes[index + 3]))
    #     ymax = int(float(classes[index + 4]))
    #
    #     class_index = (class_id - 1) * 5
    #     label[class_index] = 1
    #     label[class_index + 1] = xmin
    #     label[class_index + 2] = ymin
    #     label[class_index + 3] = xmax
    #     label[class_index + 4] = ymax
    return name, boxs


def parse_batch_data(batch_data):
    batch_name = []
    batch_box = []
    for line in batch_data:
        line = line.numpy().decode(encoding="utf-8")
        line = line.strip().split(" ")

        name, box = __parse_singal_line(line)
        batch_name.append(name)
        batch_box.append(box)

    return batch_name, batch_box


def __resize_image(image):
    """
    each picture size is different , need to resize the size of picture to uniform size for the model train

    """
    image = tf.image.resize_with_pad(image, config.IMAGE_HEIGHT, config.IMAGE_WIDTH)
    return image


def read_image_from_names(image_names):
    """
    read the image data from the specify image_path
    """
    batch_images = []
    for name in image_names:
        image = tf.io.read_file(config.JPEG_IMAGES_PATH + name)
        image = tf.io.decode_image(image, channels=config.IMAGE_CHANNEL, dtype=tf.dtypes.float32)

        image = __resize_image(image)
        batch_images.append(image)

    return tf.stack(batch_images, axis=0)


def generate_label_from_box(batch_boxs):
    """
    generate the train and test label for each picture, and we need to transform the batch_box to batch_label
    return y_label with shape (batch_size, Grid_size, Grid_size, classes_num + 5)

    """
    batch_boxs = np.asarray(batch_boxs)
    batch_label = np.zeros(shape=(config.BATCH_SIZE, config.GRID_SIZE, config.GRID_SIZE, len(config.PASCAL_VOC_CLASSES) + 5))

    x_centers = (batch_boxs[..., 1] + batch_boxs[..., 2]) // 2
    y_centers = (batch_boxs[..., 3] + batch_boxs[..., 4]) // 2

    wp_box = (batch_boxs[..., 2] - batch_boxs[..., 1]) / config.IMAGE_WIDTH
    hp_box = (batch_boxs[..., 4] - batch_boxs[..., 3]) / config.IMAGE_HEIGHT

    batch_mask = batch_boxs[..., 0] != 0  # class_id !=0 means there have box

    for i in range(config.BATCH_SIZE):
        per_mask = batch_mask[i]  # picture shape =[max_num_box_per_image, 5]
        per_picture = batch_boxs[i]
        per_pbox = per_picture[per_mask]

        per_x_center = x_centers[i][per_mask]
        per_y_center = y_centers[i][per_mask]
        per_wp_box = wp_box[i][per_mask]
        per_hp_box = hp_box[i][per_mask]

        for j in range(1):
            classes_id = int(per_pbox[j, 0])
            x = np.floor(per_x_center[j] / config.IMAGE_WIDTH * config.GRID_SIZE).astype(int)
            y = np.floor(per_y_center[j] / config.IMAGE_HEIGHT * config.GRID_SIZE).astype(int)

            batch_label[i, y, x, 0] = 1  # label: 1 or 0
            batch_label[i, y, x, classes_id + 4] = 1
            batch_label[i, y, x, 3] = per_wp_box[j]  # width
            batch_label[i, y, x, 4] = per_hp_box[j]  # height

            cell_size = config.IMAGE_WIDTH / config.GRID_SIZE
            coordinate_x = (per_x_center[j] - x * cell_size) / cell_size
            coordinate_y = (per_y_center[j] - y * cell_size) / cell_size

            batch_label[i, y, x, 1] = coordinate_x
            batch_label[i, y, x, 2] = coordinate_y

    return batch_label


if __name__ == '__main__':
    get_batch_data()