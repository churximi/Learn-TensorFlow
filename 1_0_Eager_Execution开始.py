#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
功能：Get Started with Eager Execution
网址：https://www.tensorflow.org/get_started/eager
时间：2018年04月19日16:43:01
"""

import tensorflow as tf
import tensorflow.contrib.eager as tfe
from data_utils import data_transfer
from utils import evaluate, predict, draw_loss_accuracy

tf.enable_eager_execution()  # 启动eager_execution

print("TensorFlow version: {}".format(tf.VERSION))
print("Eager execution: {}".format(tf.executing_eagerly()))


class IC_Model:
    """鸢尾花分类模型"""

    def __init__(self):
        self.train_path = "data/iris_training.csv"
        self.test_path = "data/iris_test.csv"

        self.train_dataset = data_transfer(self.train_path)
        self.test_dataset = data_transfer(self.test_path)

        self.num_epochs = 201
        self.learning_rate = 0.01
        self.create_model()

    def create_model(self):
        """构建模型"""
        self.model = tf.keras.Sequential([
            tf.keras.layers.Dense(10, activation="relu", input_shape=(4,)),  # input shape required
            tf.keras.layers.Dense(10, activation="relu"),
            tf.keras.layers.Dense(3)
        ])

    def compute_loss(self, x, y):
        y_ = self.model(x)
        return tf.losses.sparse_softmax_cross_entropy(labels=y, logits=y_)

    def grad(self, x, y):
        with tfe.GradientTape() as tape:
            self.loss_value = self.compute_loss(x, y)
        return tape.gradient(self.loss_value, self.model.variables)

    def train_on_epoch(self):
        self.epoch_loss_avg = tfe.metrics.Mean()
        self.epoch_accuracy = tfe.metrics.Accuracy()

        for x, y in tfe.Iterator(self.train_dataset):
            grads = self.grad(x, y)
            self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)  # 优化器
            self.optimizer.apply_gradients(zip(grads, self.model.variables),
                                           global_step=tf.train.get_or_create_global_step())

            self.epoch_loss_avg(self.loss_value)  # add current batch loss
            self.epoch_accuracy(tf.argmax(self.model(x), axis=1, output_type=tf.int32), y)  # 计算准确率

    def train_epoches(self):
        self.train_losses = []
        self.train_accuracies = []

        for epoch in range(self.num_epochs):
            self.train_on_epoch()
            epoch_loss = self.epoch_loss_avg.result()
            epoch_accuracy = self.epoch_accuracy.result()

            self.train_losses.append(epoch_loss)
            self.train_accuracies.append(epoch_accuracy)

            if epoch % 50 == 0:
                print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch, epoch_loss, epoch_accuracy))


def main():
    ic_model = IC_Model()
    # 绘图
    draw_loss_accuracy(ic_model.train_losses, ic_model.train_accuracies)
    # 在测试集上评估
    evaluate(ic_model.model, ic_model.test_dataset)
    # 预测未知数据
    predict(ic_model.model)


if __name__ == "__main__":
    main()
