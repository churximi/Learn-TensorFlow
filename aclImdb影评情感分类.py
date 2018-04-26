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
    return [x["class_ids"][0] for x in estimator.predict(input_fn=input_fn)]


def main():
    tf.logging.set_verbosity(tf.logging.INFO)  # Reduce logging output.

    train_df, test_df = load_datasets()
    print(train_df.head())

    # Training input on the whole training set with no limit on training epochs.
    train_input_fn = tf.estimator.inputs.pandas_input_fn(
        train_df, train_df["polarity"], num_epochs=None, shuffle=True)

    # Prediction on the whole training set.
    predict_train_input_fn = tf.estimator.inputs.pandas_input_fn(
        train_df, train_df["polarity"], shuffle=False)
    # Prediction on the test set.
    predict_test_input_fn = tf.estimator.inputs.pandas_input_fn(
        test_df, test_df["polarity"], shuffle=False)

    print("embedding")
    embedded_text_feature_column = hub.text_embedding_column(
        key="sentence",
        module_spec="/Users/simon/Downloads/tfhub_modules")  # 提前下载保存的tfhub_modules

    print("estimator")
    estimator = tf.estimator.DNNClassifier(
        hidden_units=[500, 100],
        feature_columns=[embedded_text_feature_column],
        n_classes=2,
        model_dir="models/aclImdb",
        optimizer=tf.train.AdagradOptimizer(learning_rate=0.003))

    # Training for 1,000 steps means 128,000 training examples with the default
    # batch size. This is roughly equivalent to 5 epochs since the training dataset
    # contains 25,000 examples.
    print("train...")
    estimator.train(input_fn=train_input_fn, steps=1000)

    train_eval_result = estimator.evaluate(input_fn=predict_train_input_fn)
    test_eval_result = estimator.evaluate(input_fn=predict_test_input_fn)

    print("Training set accuracy: {accuracy}".format(**train_eval_result))
    print("Test set accuracy: {accuracy}".format(**test_eval_result))

    LABELS = ["negative", "positive"]

    # Create a confusion matrix on training data.
    with tf.Graph().as_default():
        cm = tf.confusion_matrix(train_df["polarity"],
                                 get_predictions(estimator, predict_train_input_fn))
        with tf.Session() as session:
            cm_out = session.run(cm)

    # Normalize the confusion matrix so that each row sums to 1.
    cm_out = cm_out.astype(float) / cm_out.sum(axis=1)[:, np.newaxis]

    sns.heatmap(cm_out, annot=True, xticklabels=LABELS, yticklabels=LABELS)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()


if __name__ == "__main__":
    main()
