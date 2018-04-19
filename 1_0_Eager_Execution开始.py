#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
功能：Get Started with Eager Execution
网址：https://www.tensorflow.org/get_started/eager
时间：2018年04月19日15:37:37
"""

import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow.contrib.eager as tfe

tf.enable_eager_execution()  # 启动eager_execution

print("TensorFlow version: {}".format(tf.VERSION))
print("Eager execution: {}".format(tf.executing_eagerly()))


def parse_csv(line):
    example_defaults = [[0.], [0.], [0.], [0.], [0]]  # 设置字段类型
    parsed_line = tf.decode_csv(line, example_defaults)
    features = tf.reshape(parsed_line[:-1], shape=(4,))  # 抽取前四列特征
    label = tf.reshape(parsed_line[-1], shape=())  # 抽取最后一列标签
    return features, label


def data_transfer(data_path):
    """生成数据集"""
    dataset = tf.data.TextLineDataset(data_path)
    dataset = dataset.skip(1)  # skip the first header row
    dataset = dataset.map(parse_csv)  # parse each row
    dataset = dataset.shuffle(buffer_size=1000)  # randomize
    dataset = dataset.batch(32)

    # features, label = tfe.Iterator(dataset).next()  # 查看单个数据
    return dataset


def get_dataset():
    train_path = "data/iris_training.csv"
    test_path = "data/iris_test.csv"
    train_dataset = data_transfer(train_path)
    test_dataset = data_transfer(test_path)

    return train_dataset, test_dataset


def loss(model, x, y):
    y_ = model(x)
    return tf.losses.sparse_softmax_cross_entropy(labels=y, logits=y_)


def grad(model, inputs, targets):
    with tfe.GradientTape() as tape:
        loss_value = loss(model, inputs, targets)
    return tape.gradient(loss_value, model.variables)


def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, activation="relu", input_shape=(4,)),  # input shape required
        tf.keras.layers.Dense(10, activation="relu"),
        tf.keras.layers.Dense(3)
    ])
    return model


def train(train_dataset, model, optimizer):
    train_loss_results = []
    train_accuracy_results = []

    num_epochs = 201

    for epoch in range(num_epochs):
        epoch_loss_avg = tfe.metrics.Mean()
        epoch_accuracy = tfe.metrics.Accuracy()

        for x, y in tfe.Iterator(train_dataset):
            grads = grad(model, x, y)
            optimizer.apply_gradients(zip(grads, model.variables),
                                      global_step=tf.train.get_or_create_global_step())

            epoch_loss_avg(loss(model, x, y))  # add current batch loss
            epoch_accuracy(tf.argmax(model(x), axis=1, output_type=tf.int32), y)  # 计算准确率

        # end epoch
        train_loss_results.append(epoch_loss_avg.result())
        train_accuracy_results.append(epoch_accuracy.result())

        if epoch % 50 == 0:
            print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch,
                                                                        epoch_loss_avg.result(),
                                                                        epoch_accuracy.result()))

    return train_loss_results, train_accuracy_results


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


def predicte(model):
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


def main():
    # 数据
    train_dataset, test_dataset = get_dataset()

    # 模型
    model = create_model()

    # 优化器
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)

    # 训练
    train_loss_results, train_accuracy_results = train(train_dataset, model, optimizer)

    # 绘图
    draw_loss_accuracy(train_loss_results, train_accuracy_results)

    # 在测试集上评估
    evaluate(model, test_dataset)

    # 预测位置数据
    predicte(model)


if __name__ == "__main__":
    main()
