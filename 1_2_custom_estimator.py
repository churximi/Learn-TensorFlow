#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
功能：自定义 Estimator，用于鸢尾花分类问题
时间：2018年04月20日15:29:40
"""

import tensorflow as tf
from iris_data import load_data, train_input_fn, eval_input_fn, SPECIES

batch_size = 100
train_steps = 1000


def my_model(features, labels, mode, params):
    """
    三层全连接DNN, 每层的dropout probability ：0.1
    :param features: 输入函数特征
    :param labels: 输入函数标签
    :param mode: tf.estimator.ModeKeys的一个实例，表示调用程序是请求训练、预测还是评估
    :param params: 参数
    :return:
    """

    # 设计网络层
    net = tf.feature_column.input_layer(features, params['feature_columns'])  # 输入层，输入数据，feature_columns

    for units in params['hidden_units']:  # 两个隐层
        net = tf.layers.dense(net, units=units, activation=tf.nn.relu)  # 单元数，激活函数

    # 计算 logits
    logits = tf.layers.dense(net, units=params['n_classes'], activation=None)  # 输出层，不使用激活函数

    # 最大预测值所在的位置，这里即为类别编号
    predicted_classes = tf.argmax(logits, 1)

    # ----------------
    if mode == tf.estimator.ModeKeys.PREDICT:  # 如果是预测模式
        predictions = {
            'class_ids': predicted_classes[:, tf.newaxis],  # 增加一个维度，如[1, 2] 变为[[1], [2]]
            'probabilities': tf.nn.softmax(logits),  # 计算概率
            'logits': logits,
        }
        return tf.estimator.EstimatorSpec(mode, predictions=predictions)

    # 计算loss
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

    # 评估（定义指标）
    accuracy = tf.metrics.accuracy(labels=labels,
                                   predictions=predicted_classes,
                                   name='acc_op')

    # 创建包含指标的字典
    metrics = {'accuracy': accuracy}
    tf.summary.scalar('accuracy', accuracy[1])

    # ----------------
    if mode == tf.estimator.ModeKeys.EVAL:  # 如果是评估
        return tf.estimator.EstimatorSpec(mode, loss=loss, eval_metric_ops=metrics)

    # ----------------
    # 否则是训练
    assert mode == tf.estimator.ModeKeys.TRAIN

    optimizer = tf.train.AdagradOptimizer(learning_rate=0.1)  # 优化器
    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
    return tf.estimator.EstimatorSpec(mode, loss=loss, train_op=train_op)


def main():
    # 加载数据
    (train_x, train_y), (test_x, test_y) = load_data()

    # 定义特征列
    my_feature_columns = []
    for key in train_x.keys():
        my_feature_columns.append(tf.feature_column.numeric_column(key=key))

    # 自定义模型
    classifier = tf.estimator.Estimator(
        model_fn=my_model,
        params={
            'feature_columns': my_feature_columns,
            'hidden_units': [10, 10],
            'n_classes': 3,
        })

    # 训练模型，调用train()方法时，Estimator 框架会调用模型函数并将 mode 设为 ModeKeys.TRAIN
    classifier.train(
        input_fn=lambda: train_input_fn(train_x, train_y, batch_size),  # 输入函数
        steps=train_steps)

    # 评估模型
    eval_result = classifier.evaluate(
        input_fn=lambda: eval_input_fn(test_x, test_y, batch_size))

    print('Test set accuracy: {:0.3f}'.format(eval_result["accuracy"]))

    # 预测
    expected = ['Setosa', 'Versicolor', 'Virginica']
    predict_x = {
        'SepalLength': [5.1, 5.9, 6.9],
        'SepalWidth': [3.3, 3.0, 3.1],
        'PetalLength': [1.7, 4.2, 5.4],
        'PetalWidth': [0.5, 1.5, 2.1],
    }

    predictions = classifier.predict(
        input_fn=lambda: eval_input_fn(predict_x,
                                       labels=None,
                                       batch_size=batch_size))

    for pred_dict, expec in zip(predictions, expected):
        template = 'Prediction is "{}" ({:.1f}%), expected "{}"'

        class_id = pred_dict['class_ids'][0]
        probability = pred_dict['probabilities'][class_id]

        print(template.format(SPECIES[class_id], 100 * probability, expec))


if __name__ == '__main__':
    # tf.logging.set_verbosity(tf.logging.INFO)
    main()
