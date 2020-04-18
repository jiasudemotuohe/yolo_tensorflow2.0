# -*- coding: utf-8 -*-
# @Time    : 2020-04-18 13:33
# @Author  : speeding_motor

import os
import config
import xml.dom.minidom
import time


def __xml_parse(xml_file):
    file_path = os.path.join(config.Annotations_PATH, xml_file)
    dom_tree = xml.dom.minidom.parse(file_path)

    collection = dom_tree.documentElement

    file_name_node = collection.getElementsByTagName('filename')
    file_name = file_name_node[0].childNodes[0].data

    sizes = collection.getElementsByTagName('size')

    for size in sizes:
        height = size.getElementsByTagName('height')[0].childNodes[0].data
        width = size.getElementsByTagName('width')[0].childNodes[0].data
        channel = size.getElementsByTagName('depth')[0].childNodes[0].data

    objects = collection.getElementsByTagName('object')

    obj_box_lists = []

    for object in objects:
        class_name = object.getElementsByTagName('name')[0].childNodes[0].data

        box = object.getElementsByTagName('bndbox')[0]
        xmin = box.getElementsByTagName('xmin')[0].childNodes[0].data
        ymin = box.getElementsByTagName('ymin')[0].childNodes[0].data
        xmax = box.getElementsByTagName('xmax')[0].childNodes[0].data
        ymax = box.getElementsByTagName('ymax')[0].childNodes[0].data

        """
        each picture height and width in payscalvoc is different ,we need to change to unifrom size
        """
        xmin, ymin, xmax, ymax = __relocation_box_with_pad(height, width, xmin, ymin, xmax, ymax)

        obj_box_lists.append([config.PASCAL_VOC_CLASSES[class_name], xmin, ymin, xmax, ymax])

    return file_name, obj_box_lists


def __relocation_box_with_pad(height, width, xmin, ymin, xmax, ymax):
    """
    notation:
            here wo need to convert the str to float, in case of the original str contains the '.'
    """
    height = int(float(height))
    width = int(float(width))
    xmin = int(float(xmin))
    ymin = int(float(ymin))
    xmax = int(float(xmax))
    ymax = int(float(ymax))

    if height <= width:
        scale = config.IMAGE_WIDTH / width

        padding = (config.IMAGE_HEIGHT - height * scale) / 2
        xmin = xmin * scale
        xmax = xmax * scale
        ymin = ymin * scale + padding
        ymax = ymax * scale + padding

    else:
        scale = config.IMAGE_HEIGHT / height

        padding = (config.IMAGE_WIDTH - width * scale) / 2
        xmin = xmin * scale + padding
        xmax = xmax * scale + padding
        ymin = ymin * scale
        ymax = ymax * scale

    return int(xmin), int(ymin), int(xmax), int(ymax)


def __combine_pictureinfo_to_line(file_name, obj_box_lists):
    file_name += " "
    for obj in obj_box_lists:
        for item in obj:
            file_name += str(item) + " "
    return file_name


def parse_pascal_voc_to_txt():
    """
    parse the pascal xml annotation to txt file, if the pascal, if txt file is exist, delete it first

    """
    if os.path.exists(config.PASCAL_TXT_FILE):
        os.remove(config.PASCAL_TXT_FILE)

    files = os.listdir(config.Annotations_PATH)

    for file in files:
        file_name, obj_box_lists = __xml_parse(file)
        # print(file_name, obj_box_lists)

        line_info = __combine_pictureinfo_to_line(file_name, obj_box_lists)
        with open(config.PASCAL_TXT_FILE, mode="a") as f:
            f.write(line_info + "\n")


if __name__ == '__main__':
    parse_pascal_voc_to_txt()

    print("parse pascal image spend %s second" % time.clock())