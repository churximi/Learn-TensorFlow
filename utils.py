#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
功能：功能模块
时间：2018年04月19日16:44:13
"""
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.contrib.eager as tfe


def draw_loss_accuracy(train_loss, train_accuracy):
    fig, axes = plt.subplots(2, sharex=True, figsize=(12, 8))
    fig.suptitle('Training Metrics')

    axes[0].set_ylabel("Loss", fontsize=14)
    axes[0].plot(train_loss)

    axes[1].set_ylabel("Accuracy", fontsize=14)
    axes[1].set_xlabel("Epoch", fontsize=14)
    axes[1].plot(train_accuracy)

    plt.show()


def evaluate(model, test_dataset):
    test_accuracy = tfe.metrics.Accuracy()

    for (x, y) in tfe.Iterator(test_dataset):
        prediction = tf.argmax(model(x), axis=1, output_type=tf.int32)
        test_accuracy(prediction, y)

    print("Test set accuracy: {:.3%}".format(test_accuracy.result()))


def predict(model):
    class_ids = ["Iris setosa", "Iris versicolor", "Iris virginica"]

    predict_dataset = tf.convert_to_tensor([
        [5.1, 3.3, 1.7, 0.5, ],
        [5.9, 3.0, 4.2, 1.5, ],
        [6.9, 3.1, 5.4, 2.1]
    ])

    predictions = model(predict_dataset)

    for i, logits in enumerate(predictions):
        class_idx = tf.argmax(logits).numpy()
        name = class_ids[class_idx]
        print("Example {} prediction: {}".format(i, name))


if __name__ == "__main__":
    pass
