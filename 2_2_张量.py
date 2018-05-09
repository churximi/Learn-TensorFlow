# !/usr/bin/env python
# -*- coding: utf-8 -*-

"""
功能：张量（tensor）相关内容
时间：2018年05月09日09:04:19
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tensorflow as tf


def rank_0():
    """0阶"""
    mammal = tf.Variable("Elephant", tf.string)
    ignition = tf.Variable(451, tf.int16)
    floating = tf.Variable(3.14159265359, tf.float64)
    its_complicated = tf.Variable(12.3 - 4.85j, tf.complex64)
    print(mammal.shape)


def rank_1():
    """1阶"""
    mystr = tf.Variable(["Hello"], tf.string)
    cool_numbers = tf.Variable([3.14159, 2.71828], tf.float32)
    first_primes = tf.Variable([2, 3, 5, 7, 11], tf.int32)
    its_very_complicated = tf.Variable([12.3 - 4.85j, 7.5 - 6.23j], tf.complex64)
    print(first_primes.shape)


def rank_n():
    """n阶"""
    mymat = tf.Variable([[7], [11]], tf.int16)
    myxor = tf.Variable([[False, True], [True, False]], tf.bool)
    linear_squares = tf.Variable([[4], [9], [16], [25]], tf.int32)
    squarish_squares = tf.Variable([[4, 9], [16, 25]], tf.int32)
    rank_of_squares = tf.rank(squarish_squares)
    mymatC = tf.Variable([[7], [11]], tf.int32)
    print("squarish_squares的形状：{}".format(squarish_squares.shape))
    with tf.Session() as sess:
        print("squarish_squares的阶数：{}".format(sess.run(rank_of_squares)))


def tensor_reshape():
    """形状及转换"""
    rank_three_tensor = tf.ones([3, 4, 5])
    matrixA = tf.reshape(rank_three_tensor, [6, 10])  # 改为（6，10）
    matrixB = tf.reshape(matrixA, [3, -1])  # 改为（3，20），-1表示自动计算该维度
    matrixC = tf.reshape(matrixB, [4, 3, -1])  # 改为（4，3，5）
    # yet_another = tf.reshape(matrixC, [13, 2, -1])  # ERROR，不匹配

    print(matrixA.shape)
    print(matrixB.shape)
    print(matrixC.shape)


def data_type():
    """数据类型及转换"""
    # Cast a constant integer tensor into floating point.
    x = tf.constant([1, 2, 3])
    y = tf.cast(x, dtype=tf.float32)  # 转换
    print(x.dtype, y.dtype)


def evaluate_tensor():
    """评估张量"""
    constant = tf.constant([1, 2, 3])
    tensor = constant * constant

    p = tf.placeholder(tf.float32)
    t = p + 1.0

    with tf.Session() as sess:
        print(tensor.eval())
        print(t.eval(feed_dict={p: 2.0}))  # 需要给占位符提供值，才可以调用eval


def print_tensor():
    """打印张量，tf.Print的使用"""
    t = tf.constant([1, 2, 3])
    tf.Print(t, [t])  # This does nothing
    t = tf.Print(t, [t])  # tf.Print返回的结果
    result = t + 1
    with tf.Session() as sess:
        print("result：", result.eval())  # 在评估result时才会打印t


def main():
    rank_0()
    rank_1()
    rank_n()
    tensor_reshape()
    data_type()
    evaluate_tensor()
    print_tensor()


if __name__ == "__main__":
    main()
