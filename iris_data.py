#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
功能：鸢尾花数据处理
时间：2018年04月20日14:52:44
"""

import pandas as pd
import tensorflow as tf

CSV_COLUMN_NAMES = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']  # 列名
SPECIES = ['Setosa', 'Versicolor', 'Virginica']  # 类别
train_path = "data/iris_training.csv"
test_path = "data/iris_test.csv"


def load_data(y_name='Species'):
    """ 返回数据集格式 (train_x, train_y), (test_x, test_y)"""
    train = pd.read_csv(train_path, names=CSV_COLUMN_NAMES, header=0)  # 指定列名
    train_x, train_y = train, train.pop(y_name)

    test = pd.read_csv(test_path, names=CSV_COLUMN_NAMES, header=0)
    test_x, test_y = test, test.pop(y_name)

    return (train_x, train_y), (test_x, test_y)


def train_input_fn(features, labels, batch_size):
    """ 训练用的输入函数"""
    # 将输入转换为Dataset
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))  # dict转换

    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)

    return dataset


def eval_input_fn(features, labels, batch_size):
    """ 评估/预测用的输入函数"""
    features = dict(features)  # dict转换
    if labels is None:  # 预测时没有label
        inputs = features
    else:  # test有label
        inputs = (features, labels)

    dataset = tf.data.Dataset.from_tensor_slices(inputs)

    # Batch the examples
    assert batch_size is not None, "batch_size must not be None"
    dataset = dataset.batch(batch_size)  # 评估/预测时不需要shuffle和repeat

    return dataset


# 下面是一个csv解析器的示例,使用Dataset类实现

# `tf.parse_csv` sets the types of the outputs to match the examples given in
#     the `record_defaults` argument.
CSV_TYPES = [[0.0], [0.0], [0.0], [0.0], [0]]


def _parse_line(line):
    # Decode the line into its fields
    fields = tf.decode_csv(line, record_defaults=CSV_TYPES)

    # Pack the result into a dictionary
    features = dict(zip(CSV_COLUMN_NAMES, fields))

    # Separate the label from the features
    label = features.pop('Species')

    return features, label


def csv_input_fn(csv_path, batch_size):
    # Create a dataset containing the text lines.
    dataset = tf.data.TextLineDataset(csv_path).skip(1)

    # Parse each line.
    dataset = dataset.map(_parse_line)

    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)

    # Return the dataset.
    return dataset


if __name__ == "__main__":
    pass
