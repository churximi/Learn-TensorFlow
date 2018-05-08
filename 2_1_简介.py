# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
功能：低级别API_简介
时间：2018年05月08日09:23:12
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import tensorflow as tf


def learn_01():
    graph = tf.Graph()
    with graph.as_default():
        a = tf.constant(3.0, dtype=tf.float32)
        b = tf.constant(4.0)  # also tf.float32 implicitly
        total = a + b
        print(a)
        print(b)
        print(total)

        writer = tf.summary.FileWriter('summaries')
        writer.add_graph(tf.get_default_graph())

        with tf.Session() as sess:
            print(sess.run(total))
            print(sess.run({'a和b': (a, b), 'total': total}))
            z = sess.run({'a': a, 'b': b, 'total': total})
            print(z, type(z))


def learn_02():
    # Constant 1-D Tensor populated with value list.
    tensor_01 = tf.constant([1, 2, 3, 4, 5, 6, 7])
    print(tensor_01)

    # Constant 2-D tensor populated with scalar value -1.
    tensor_02 = tf.constant([-1.0, 2.0], shape=[2, 3])
    print(tensor_02)

    with tf.Session() as sess:
        print(sess.run(tensor_01))
        print(sess.run(tensor_02))


def learn_03():
    """占位符"""
    x = tf.placeholder(tf.float32)
    y = tf.placeholder(tf.float32)
    z = x + y

    with tf.Session() as sess:
        print(sess.run(z, feed_dict={x: 3, y: 4.5}))
        print(sess.run(z, feed_dict={x: [1, 3], y: [2, 4]}))


def learn_04():
    """数据集"""
    my_data = [
        [0, 1, ],
        [2, 3, ],
        [4, 5, ],
        [6, 7, ],
    ]
    slices = tf.data.Dataset.from_tensor_slices(my_data)
    next_item = slices.make_one_shot_iterator().get_next()

    with tf.Session() as sess:
        while True:
            try:
                print(sess.run(next_item))
            except tf.errors.OutOfRangeError:
                break


def learn_05():
    """层"""
    x = tf.placeholder(tf.float32, shape=[None, 3])
    linear_model = tf.layers.Dense(units=1)
    y = linear_model(x)

    with tf.Session() as sess:
        init = tf.global_variables_initializer()  # 初始化层
        sess.run(init)
        print(sess.run(y, feed_dict={x: [[1, 2, 3], [4, 5, 6]]}))


def small_lr():
    """简单的回归模型"""
    # 数据
    x = tf.constant([[1], [2], [3], [4]], dtype=tf.float32)
    y_true = tf.constant([[0], [-1], [-2], [-3]], dtype=tf.float32)

    # 模型
    linear_model = tf.layers.Dense(units=1)
    y_pred = linear_model(x)

    # 评估
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)

        print(sess.run(y_pred))

        # 损失
        loss = tf.losses.mean_squared_error(labels=y_true, predictions=y_pred)
        print(sess.run(loss))

        # 优化训练
        optimizer = tf.train.GradientDescentOptimizer(0.01)  # 梯度下降
        train = optimizer.minimize(loss)
        for i in range(100):
            _, loss_value = sess.run((train, loss))
            if i % 10 == 0:
                print(loss_value)

        print(sess.run(y_pred))


def main():
    small_lr()


if __name__ == "__main__":
    main()
