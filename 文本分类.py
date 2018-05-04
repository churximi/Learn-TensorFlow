# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
功能：中文文本分类
时间：2018年05月04日10:42:29
数量：训练集：121155，测试集：103369
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import re
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub


def preprocess_data(file_path):
    pat = re.compile("(.*?)__label__(.*?)\n")
    data = dict()
    data["sentence"] = []
    data["category"] = []

    labels = ['stock', 'edu', 'ent', 'constellation', 'affairs', 'house', 'home',
              'lottery', 'economic', 'game', 'science', 'sports', 'fashion']
    label_id = {l: labels.index(l) for l in labels}

    with open(file_path) as f:
        for index, line in enumerate(f):
            finds = pat.findall(line)
            text = finds[0][0].rstrip()
            text = text.replace("\u3000", "")
            while "  " in text:
                text = text.replace("  ", " ")
            data["sentence"].append(text)
            data["category"].append(label_id[finds[0][1]])

    print("数据集数量：", len(data["sentence"]))
    assert len(data["sentence"]) == len(data["category"])

    return pd.DataFrame.from_dict(data)


def main():
    tf.logging.set_verbosity(tf.logging.INFO)

    train_df = preprocess_data("data/news_fasttext_train.txt")
    test_df = preprocess_data("data/news_fasttext_test.txt")
    print(train_df.head())
    print(test_df.head())

    # 定义输入函数
    # （1）Training input on the whole training set with no limit on training epochs.
    train_input_fn = tf.estimator.inputs.pandas_input_fn(
        train_df, train_df["category"], num_epochs=None, shuffle=True)  # 随机化

    # （2）整个训练集上预测
    predict_train_input_fn = tf.estimator.inputs.pandas_input_fn(
        train_df, train_df["category"], shuffle=False)
    # （3）测试集上预测
    predict_test_input_fn = tf.estimator.inputs.pandas_input_fn(
        test_df, test_df["category"], shuffle=False)

    # 定义特征列
    print("embedding")
    embedded_text_feature_column = hub.text_embedding_column(
        key="sentence",
        module_spec="/Users/simon/Downloads/tfhub_modules_zh")
    # module_spec="/Users/simon/Downloads/tfhub_modules"  # 提前下载保存的tfhub_modules

    # 选择模型
    print("estimator")
    estimator = tf.estimator.DNNClassifier(
        feature_columns=[embedded_text_feature_column],
        hidden_units=[256, 128],
        n_classes=13,
        model_dir="models/news",
        optimizer=tf.train.AdagradOptimizer(learning_rate=0.001))

    # 训练
    # 训练1,000步表示 128,000 个训练实例（默认batch size为128），约等于 5 轮（128000/25000）
    print("train...")
    estimator.train(input_fn=train_input_fn, steps=10000)

    # 评估
    train_eval_result = estimator.evaluate(input_fn=predict_train_input_fn)
    test_eval_result = estimator.evaluate(input_fn=predict_test_input_fn)

    print("Training set accuracy: {accuracy}".format(**train_eval_result))
    print("Test set accuracy: {accuracy}".format(**test_eval_result))


if __name__ == "__main__":
    main()
