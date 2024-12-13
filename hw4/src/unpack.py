# -*- coding: utf-8 -*-

import os
import struct
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

__author__ = 'zj'

# MNIST文件目录
data_dir = ".\datasets\MNIST"
# 结果文件名
result_file_name = 'decompress_mnist'

train = 'train'
test = 'test'
train_image_file = 'train-images.idx3-ubyte'
train_label_file = 'train-labels.idx1-ubyte'
test_image_file = 't10k-images.idx3-ubyte'
test_label_file = 't10k-labels.idx1-ubyte'


def decompress_label(label_file_path):
    """
    解压标签文件
    :param label_file_path: 标签文件路径
    :return: list
    """
    with open(label_file_path, 'rb') as f:
        magic_number = int.from_bytes(f.read(4), byteorder='big')
        image_number = int.from_bytes(f.read(4), byteorder='big')
        # print(magic_number)
        # print(image_number)
        fmt = '>B'
        byte_num = 1
        label_list = []
        for i in range(image_number):
            label_list.append(struct.unpack(fmt, f.read(byte_num))[0])
        # print(number_list)
        return label_list


def decompress_image(image_file_path):
    """
    解压图像文件
    :param image_file_path: 图像文件路径
    :return:
    """
    with open(image_file_path, 'rb') as f:
        magic_number = int.from_bytes(f.read(4), byteorder='big')
        image_number = int.from_bytes(f.read(4), byteorder='big')
        height = int.from_bytes(f.read(4), byteorder='big')
        width = int.from_bytes(f.read(4), byteorder='big')
        print('number: %d height: %d  width: %d' % (image_number, height, width))
        img_len = height * width
        fmt = '>' + str(img_len) + 'B'

        image_list = []
        for i in range(image_number):
            img = struct.unpack(fmt, f.read(img_len))
            img = np.reshape(img, (height, width))
            image_list.append(img)
        return image_list


def decompress_image_save(image_file_path, label_list, res_dir):
    """
    解压图像文件并保存
    :param image_file_path: 图像文件路径
    :param label_list: 标签列表
    :param res_dir: 结果路径
    :return:
    """
    with open(image_file_path, 'rb') as f:
        magic_number = int.from_bytes(f.read(4), byteorder='big')
        image_number = int.from_bytes(f.read(4), byteorder='big')
        height = int.from_bytes(f.read(4), byteorder='big')
        width = int.from_bytes(f.read(4), byteorder='big')
        print('number: %d height: %d  width: %d' % (image_number, height, width))
        img_len = height * width
        fmt = '>' + str(img_len) + 'B'

        nums = [0 for i in range(10)]
        for i in range(image_number):
            img = struct.unpack(fmt, f.read(img_len))
            img = np.reshape(img, (height, width))

            image_dir = os.path.join(res_dir, str(label_list[i]))
            if not os.path.exists(image_dir):
                os.mkdir(image_dir)
            image_path = os.path.join(image_dir, str(nums[label_list[i]]) + ".png")
            cv.imwrite(image_path, img)
            nums[label_list[i]] += 1


def decompress():
    """
    解压图像和标签文件
    :return:
    """
    # 创建结果文件路径
    des_dir = os.path.join(data_dir, result_file_name)
    if not os.path.exists(des_dir):
        os.mkdir(des_dir)
    des_train_dir = os.path.join(des_dir, train)
    if not os.path.exists(des_train_dir):
        os.mkdir(des_train_dir)
    des_test_dir = os.path.join(des_dir, test)
    if not os.path.exists(des_test_dir):
        os.mkdir(des_test_dir)
    print('mkdir result dir ok')

    label_list = decompress_label(os.path.join(data_dir, train_label_file))
    decompress_image_save(os.path.join(data_dir, train_image_file), label_list, des_train_dir)
    print('load image and label ok')

    label_list = decompress_label(os.path.join(data_dir, test_label_file))
    decompress_image_save(os.path.join(data_dir, test_image_file), label_list, des_test_dir)
    print('load image and label ok')


def show_image(img_list):
    # plt.figure(figsize=(10, 5))  # 设置窗口大小
    plt.figure()
    plt.suptitle('MNIST')  # 图片名称
    rows = 10
    cols = 10
    for i in range(cols):
        for j in range(rows):
            plt.subplot(rows, cols, j * cols + i + 1)
            plt.imshow(img_list[j * cols + i], cmap='gray'), plt.axis('off')
    plt.savefig('./mnit.png')
    plt.show()


if __name__ == '__main__':
    decompress()
    image_list = decompress_image(os.path.join(data_dir, test_image_file))
    show_image(image_list[:100])
