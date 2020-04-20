# -*- coding: utf-8 -*-
# @Time    : 2020-04-18 13:33
# @Author  : speeding_motor


Annotations_PATH = "/Users/anyongyi/Downloads/VOC2012/Annotations/"
JPEG_IMAGES_PATH = "/Users/anyongyi/Downloads/VOC2012/JPEGImages/"
PASCAL_TXT_FILE = 'datasets/data.txt'

IMAGE_HEIGHT = 416
IMAGE_WIDTH = 416
IMAGE_CHANNEL = 3

PROBABILITY_THRESHOLD = 0.6
IOU_THRESHOLD = 0.6  # intersection over union, when lou between box is over the num, then give up the box

BATCH_SIZE = 128
EPOCHS = 100


# the max_num each picture can have the box, it means max object the each picture have
MAX_NUM_BOXS_PER_IMAGE = 20

"""
we put GRID_SIZE * GRID_SIZE grid on the picture, then to predict each cell of the picture
"""
GRID_SIZE = 16

PASCAL_VOC_CLASSES = {"person": 1,
                      "bird": 2,
                      "cat": 3,
                      "cow": 4,
                      "dog": 5,
                      "horse": 6,
                      "sheep": 7,
                      "aeroplane": 8,
                      "bicycle": 9,
                      "boat": 10,
                      "bus": 11,
                      "car": 12,
                      "motorbike": 13,
                      "train": 14,
                      "bottle": 15,
                      "chair": 16,
                      "diningtable": 17,
                      "pottedplant": 18,
                      "sofa": 19,
                      "tvmonitor": 20}

