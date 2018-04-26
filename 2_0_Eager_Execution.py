#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
功能：Eager Execution学习
时间：2018年04月26日11:34:46
"""

import numpy as np
import tensorflow as tf
import tensorflow.contrib.eager as tfe

tf.enable_eager_execution()


def learn_01():
    tf.executing_eagerly()  # => True

    x = [[2.]]
    m = tf.matmul(x, x)
    print("hello, {}".format(m))  # => "hello, [[4.]]"
    # 如果不启动，返回：hello, Tensor("MatMul:0", shape=(1, 1), dtype=float32)


def learn_02():
    a = tf.constant([[1, 2],
                     [3, 4]])
    print(a)

    # Broadcasting support
    b = tf.add(a, 1)
    print(b)

    # Operator overloading is supported
    print(a * b)

    # Use NumPy values
    c = np.multiply(a, b)
    print(c)

    # Obtain numpy value from a tensor:
    print(a.numpy())


def learn_03():
    w = tfe.Variable([[1.0]])
    with tfe.GradientTape() as tape:
        loss = w * w

    grad = tape.gradient(loss, [w])
    print(grad)  # => [<tf.Tensor: id=31, shape=(1, 1), dtype=float32, numpy=array([[2.]], dtype=float32)>]


def learn_04():
    # A toy dataset of points around 3 * x + 2
    NUM_EXAMPLES = 1000
    training_inputs = tf.random_normal([NUM_EXAMPLES])
    noise = tf.random_normal([NUM_EXAMPLES])
    training_outputs = training_inputs * 3 + 2 + noise

    def prediction(input, weight, bias):
        return input * weight + bias

    # A loss function using mean-squared error
    def loss(weights, biases):
        error = prediction(training_inputs, weights, biases) - training_outputs
        return tf.reduce_mean(tf.square(error))

    # Return the derivative of loss with respect to weight and bias
    def grad(weights, biases):
        with tfe.GradientTape() as tape:
            loss_value = loss(weights, biases)
        return tape.gradient(loss_value, [weights, biases])

    train_steps = 200
    learning_rate = 0.01
    # Start with arbitrary values for W and B on the same batch of data
    W = tfe.Variable(5.)
    B = tfe.Variable(10.)

    print("Initial loss: {:.3f}".format(loss(W, B)))

    for i in range(train_steps):
        dW, dB = grad(W, B)  # 计算梯度
        W.assign_sub(dW * learning_rate)  # assign_sub：变量减去一个值
        B.assign_sub(dB * learning_rate)
        if i % 20 == 0:
            print("Loss at step {:03d}: {:.3f}".format(i, loss(W, B)))

    print("Final loss: {:.3f}".format(loss(W, B)))
    print("W = {}, B = {}".format(W.numpy(), B.numpy()))


def main():
    learn_01()
    learn_02()
    learn_03()
    learn_04()


if __name__ == "__main__":
    main()
