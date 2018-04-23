#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
功能：Datasets Quick Start学习，数据集快速入门
时间：2018年04月23日14:22:48
"""

import numpy as np
import tensorflow as tf
from iris_data import csv_input_fn, CSV_COLUMN_NAMES, eval_input_fn, load_data

tf.logging.set_verbosity(tf.logging.INFO)


def learn_slice():
    mnist_file_path = "/Users/simon/Mycodes/Learn-TensorFlow/data/mnist.npz"  # mnist文件所在绝对路径
    train, test = tf.keras.datasets.mnist.load_data(path=mnist_file_path)
    mnist_x, mnist_y = train

    mnist_ds = tf.data.Dataset.from_tensor_slices(mnist_x)
    print(mnist_ds)
    print(mnist_ds.batch(100))


def learn_slice2():
    features = (("SepalLength", np.array([1.0, 2.0])),
                ("PetalWidth", np.array([3.0, 4.0])),
                ("PetalLength", np.array([5.0, 6.0])),
                ("SepalWidth", np.array([7.0, 8.0])))
    dataset = tf.data.Dataset.from_tensor_slices(dict(features))
    print(dataset)
    print(dataset.batch(10))
    data_result = dataset.make_one_shot_iterator().get_next()
    print(data_result)


def my_model():
    train_path = "data/iris_training.csv"
    test_path = "data/iris_test.csv"
    batch_size = 100
    train_steps = 1000

    _, (test_x, test_y) = load_data()

    # All the inputs are numeric
    # 定义特征列
    feature_columns = [
        tf.feature_column.numeric_column(key)
        for key in CSV_COLUMN_NAMES[:-1]]

    # 构建estimator
    classifier = tf.estimator.LinearClassifier(feature_columns=feature_columns,
                                               n_classes=3,
                                               model_dir="models/iris_Linear")

    # 训练
    classifier.train(
        input_fn=lambda: csv_input_fn(train_path, batch_size),
        steps=train_steps)

    # 评估，返回eval_result是一个字典，有4个key：accuracy，average_loss，global_step，loss
    eval_result = classifier.evaluate(
        input_fn=lambda: eval_input_fn(features=test_x,
                                       labels=test_y,
                                       batch_size=batch_size))

    print('Test set accuracy: {:0.3f}'.format(eval_result["accuracy"]))


def main():
    my_model()


if __name__ == "__main__":
    pass
