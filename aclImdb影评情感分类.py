#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
功能：使用TF-Hub文本分类
时间：2018年04月26日14:24:33
"""

import tensorflow as tf
import tensorflow_hub as hub
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import json


# Load all files from a directory in a DataFrame.
def load_directory_data(directory):
    data = dict()
    data["sentence"] = []
    data["sentiment"] = []

    with open(directory) as f:
        original_data = json.load(f)

    for d in original_data:
        data["sentence"].append(d["sentence"])
        data["sentiment"].append(d["sentiment"])
    return pd.DataFrame.from_dict(data)


# Merge positive and negative examples, add a polarity column and shuffle.
def load_dataset(pos_path, neg_path):
    pos_df = load_directory_data(pos_path)
    neg_df = load_directory_data(neg_path)
    pos_df["polarity"] = 1
    neg_df["polarity"] = 0
    return pd.concat([pos_df, neg_df]).sample(frac=1).reset_index(drop=True)


def load_datasets():
    train_pos_path = "data/aclImdb/train/aclImdb_train_pos.json"
    train_neg_path = "data/aclImdb/train/aclImdb_train_neg.json"
    test_pos_path = "data/aclImdb/test/aclImdb_test_pos.json"
    test_neg_path = "data/aclImdb/test/aclImdb_test_neg.json"

    train_df = load_dataset(train_pos_path, train_neg_path)
    test_df = load_dataset(test_pos_path, test_neg_path)

    return train_df, test_df


def get_predictions(estimator, input_fn):
    predictions = []
    for x in estimator.predict(input_fn=input_fn):
        predictions.append(x["class_ids"][0])
    return predictions


def read_data(sentence):
    """新数据转换，用于预测"""
    data = dict()
    data["sentence"] = []
    for sen in sentence:
        data["sentence"].append(sen)

    return pd.DataFrame.from_dict(data)


def main():
    tf.logging.set_verbosity(tf.logging.INFO)  # Reduce logging output.

    # 加载数据
    train_df, test_df = load_datasets()  # 句子，评分(1,2,3,4,7,8,9,10)，极性(0, 1)

    # 定义输入函数
    # （1）Training input on the whole training set with no limit on training epochs.
    train_input_fn = tf.estimator.inputs.pandas_input_fn(
        train_df, train_df["polarity"], num_epochs=None, shuffle=True)  # 随机化

    # （2）整个训练集上预测
    predict_train_input_fn = tf.estimator.inputs.pandas_input_fn(
        train_df, train_df["polarity"], shuffle=False)
    # （3）测试集上预测
    predict_test_input_fn = tf.estimator.inputs.pandas_input_fn(
        test_df, test_df["polarity"], shuffle=False)

    # 定义特征列
    print("embedding")
    embedded_text_feature_column = hub.text_embedding_column(
        key="sentence",
        module_spec="/Users/simon/Downloads/tfhub_modules")  # 提前下载保存的tfhub_modules

    # 选择模型
    print("estimator")
    estimator = tf.estimator.DNNClassifier(
        feature_columns=[embedded_text_feature_column],
        hidden_units=[500, 100],
        n_classes=2,
        model_dir="models/aclImdb",
        optimizer=tf.train.AdagradOptimizer(learning_rate=0.003))

    # 训练
    # 训练1,000步表示 128,000 个训练实例（默认batch size为128），约等于 5 轮（128000/25000）
    print("train...")
    # estimator.train(input_fn=train_input_fn, steps=1000)

    # 评估
    train_eval_result = estimator.evaluate(input_fn=predict_train_input_fn)
    test_eval_result = estimator.evaluate(input_fn=predict_test_input_fn)

    print("Training set accuracy: {accuracy}".format(**train_eval_result))
    print("Test set accuracy: {accuracy}".format(**test_eval_result))

    # 预测
    LABELS = ["negative", "positive"]
    sentence = ["I like this movie very much.",
                "This movie is wasteful of talent."]
    data = read_data(sentence)
    # （4）在新数据上预测
    predict_new_input_fn = tf.estimator.inputs.pandas_input_fn(x=data,
                                                               y=None,
                                                               shuffle=False)
    predictions = estimator.predict(input_fn=predict_new_input_fn)
    for sen, pred in zip(sentence, predictions):
        class_id = pred['class_ids'][0]
        print("预测句子：{}".format(sen))
        print("预测结果：{}".format(LABELS[class_id]))

    """
    # 混淆矩阵
    # Create a confusion matrix on training data.
    with tf.Graph().as_default():
        predictions = get_predictions(estimator, predict_train_input_fn)
        cm = tf.confusion_matrix(labels=train_df["polarity"],
                                 predictions=predictions)
        with tf.Session() as sess:
            cm_out = sess.run(cm)

    # Normalize the confusion matrix so that each row sums to 1.
    cm_out = cm_out.astype(float) / cm_out.sum(axis=1)[:, np.newaxis]

    sns.heatmap(cm_out, annot=True, xticklabels=LABELS, yticklabels=LABELS)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()
    """


if __name__ == "__main__":
    main()
