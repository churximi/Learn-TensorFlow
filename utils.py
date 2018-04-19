#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
功能：
时间：
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


load_iris_data()
if __name__ == "__main__":
    pass
