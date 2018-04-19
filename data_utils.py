#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
功能：数据处理相关
时间：2018年04月19日16:41:03
"""

import os
import tensorflow as tf
import pandas as pd


def download_data():
    """下载鸢尾花数据集"""
    train_url = "http://download.tensorflow.org/data/iris_training.csv"
    test_url = "http://download.tensorflow.org/data/iris_test.csv"

    save_path = "/Users/simon/Mycodes/Learn-TensorFlow/data2"  # 指定保存路径
    for url in [train_url, test_url]:
        tf.keras.utils.get_file(fname=os.path.basename(url), origin=url, cache_subdir=save_path)


def load_iris_data():
    csv_path = "data/iris_training.csv"
    data = pd.read_csv(csv_path)
    print(data.head())
    return data


def parse_csv(line):
    example_defaults = [[0.], [0.], [0.], [0.], [0]]  # 设置字段类型
    parsed_line = tf.decode_csv(line, example_defaults)
    features = tf.reshape(parsed_line[:-1], shape=(4,))  # 抽取前四列特征
    label = tf.reshape(parsed_line[-1], shape=())  # 抽取最后一列标签
    return features, label


def data_transfer(data_path):
    """生成数据集"""
    dataset = tf.data.TextLineDataset(data_path)
    dataset = dataset.skip(1)  # skip the first header row
    dataset = dataset.map(parse_csv)  # parse each row
    dataset = dataset.shuffle(buffer_size=1000)  # randomize
    dataset = dataset.batch(32)

    # features, label = tfe.Iterator(dataset).next()  # 查看单个数据
    return dataset


if __name__ == "__main__":
    pass
