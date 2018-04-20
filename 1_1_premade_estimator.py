#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
功能：DNNClassifier，鸢尾花数据集分类
时间：2018年04月20日10:48:35
"""

import tensorflow as tf
from iris_data import load_data, train_input_fn, eval_input_fn, SPECIES

batch_size = 100
train_steps = 1000


def main():
    # 加载数据
    (train_x, train_y), (test_x, test_y) = load_data()

    # 定义特征列
    my_feature_columns = []
    for key in train_x.keys():
        my_feature_columns.append(tf.feature_column.numeric_column(key=key))

    # 选择模型
    classifier = tf.estimator.DNNClassifier(
        feature_columns=my_feature_columns,
        hidden_units=[10, 10],  # 定义每个隐层的单元数（2个隐层，每层10个隐藏单元）
        n_classes=3,  # 类别数
        model_dir="models/iris")  # 指定模型保存目录

    # 训练，传入数据train_x即为特征，train_y即为标签
    classifier.train(
        input_fn=lambda: train_input_fn(features=train_x,
                                        labels=train_y,
                                        batch_size=batch_size),
        steps=train_steps)

    # 评估，返回eval_result是一个字典，有4个key：accuracy，average_loss，global_step，loss
    eval_result = classifier.evaluate(
        input_fn=lambda: eval_input_fn(features=test_x,
                                       labels=test_y,
                                       batch_size=batch_size))

    print('Test set accuracy: {:0.3f}'.format(eval_result["accuracy"]))

    # 预测，3个实例
    expected = ['Setosa', 'Versicolor', 'Virginica']  # 这3个实例的期望类别
    predict_x = {
        'SepalLength': [5.1, 5.9, 6.9],
        'SepalWidth': [3.3, 3.0, 3.1],
        'PetalLength': [1.7, 4.2, 5.4],
        'PetalWidth': [0.5, 1.5, 2.1],
    }

    # predictions包含所有的预测
    predictions = classifier.predict(
        input_fn=lambda: eval_input_fn(features=predict_x,
                                       labels=None,
                                       batch_size=batch_size))

    template = 'Prediction is "{}" ({:.1f}%), expected "{}"'  # 类别，概率，期望类别

    # 打印预测结果
    for pred_dict, expec in zip(predictions, expected):
        class_id = pred_dict['class_ids'][0]  # 预测的标签编号
        probability = pred_dict['probabilities'][class_id]  # 该类别的概率

        print(template.format(SPECIES[class_id], 100 * probability, expec))


if __name__ == '__main__':
    # tf.logging.set_verbosity(tf.logging.INFO)
    main()
