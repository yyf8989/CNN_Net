#!/usr/bin/env python
# -*- coding:utf-8 -*-
"""
__author__ = 'YYF'
__mtime__ = '2018/11/17'
# code is far away from bugs with the god animal protecting
    I love animals. They taste delicious.
              ┏┓   ┏┓
            ┏┛┻━━━┛┻┓
           ┃   ☃   ┃
           ┃ ┳┛ ┗┳ ┃
           ┃   ┻    ┃
            ┗━┓   ┏━┛
              ┃    ┗━━━┓
               ┃ 神兽保佑 ┣┓
               ┃ 永无BUG! ┏┛
                ┗┓┓┏ ━┳┓┏┛
                  ┃┫┫  ┃┫┫
                  ┗┻┛  ┗┻┛
"""
import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
import numpy as np
import matplotlib.pyplot as plt

mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
save_path = r"E:\Pycharmprojects\CNN_Net\CNN_Net_Tensorflow\CNN_Net_Tensorflow_ckpt\1_ckpt"


class CNNNet:
    def __init__(self):
        self.x = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28, 1])
        self.y = tf.placeholder(dtype=tf.float32, shape=[None, 10])
        self.dp = tf.placeholder(dtype=tf.float32)
        with tf.name_scope('Conv1'):
            self.conv1_w = tf.Variable(tf.truncated_normal(shape=[3, 3, 1, 16], stddev=0.2))
            self.conv1_b = tf.Variable(tf.zeros([16]))

        with tf.name_scope('Conv2'):
            self.conv2_w = tf.Variable(tf.truncated_normal(shape=[3, 3, 16, 32], stddev=0.2))
            self.conv2_b = tf.Variable(tf.zeros([32]))

        with tf.name_scope('MLP1'):
            self.in_w1 = tf.Variable(tf.truncated_normal(shape=[7 * 7 * 32, 128], stddev=0.2))
            self.in_b1 = tf.Variable(tf.zeros([128]))

        with tf.name_scope('MLP2'):
            self.in_w2 = tf.Variable(tf.truncated_normal(shape=[128, 10], stddev=0.2))
            self.in_b2 = tf.Variable(tf.zeros([10]))

        self.forward()
        self.backward()
        self.Accuracy()

    def forward(self):
        with tf.name_scope('Conv1'):
            self.conv1 = tf.nn.conv2d(self.x, self.conv1_w, [1, 1, 1, 1], 'SAME') + self.conv1_b
            self.conv1_relu = tf.nn.leaky_relu(self.conv1)
            self.conv1_pool = tf.nn.max_pool(self.conv1_relu, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

        with tf.name_scope('Conv2'):
            self.conv2 = tf.nn.conv2d(self.conv1_pool, self.conv2_w, [1, 1, 1, 1], 'SAME') + self.conv2_b
            self.conv2_relu = tf.nn.leaky_relu(self.conv2)
            self.conv2_pool = tf.nn.max_pool(self.conv2_relu, [1, 2, 2, 1], [1, 2, 2, 1], padding='SAME')

        with tf.name_scope('Flatten'):
            self.flatten = tf.reshape(self.conv2_pool, shape=[-1, 7 * 7 * 32])

        with tf.name_scope('MLP1'):
            self.mlp1 = tf.matmul(self.flatten, self.in_w1) + self.in_b1
            self.mlp1_relu = tf.nn.leaky_relu(self.mlp1)

        with tf.name_scope('Dropout'):
            self.dropout = tf.nn.dropout(self.mlp1_relu, keep_prob=self.dp)

        with tf.name_scope('MLP2'):
            self.mlp2 = tf.matmul(self.dropout, self.in_w2) + self.in_b2
            self.output = tf.nn.softmax(self.mlp2)

    def backward(self):
        with tf.name_scope('Loss'):
            self.loss = tf.reduce_mean(tf.square(self.output - self.y))
            tf.summary.scalar("loss", self.loss)

        with tf.name_scope('Opt'):
            self.opt = tf.train.AdamOptimizer(0.0005).minimize(self.loss)

    def Accuracy(self):
        with tf.name_scope('Accuracy'):
            self.bool_output = tf.equal(tf.argmax(self.output, 1), tf.argmax(self.y, 1))
            self.accu = tf.reduce_mean(tf.cast(self.bool_output, dtype=tf.float32))
            tf.summary.scalar("accuracy", self.accu)


if __name__ == '__main__':
    cnn = CNNNet()
    init = tf.global_variables_initializer()

    save = tf.train.Saver()
    merged = tf.summary.merge_all()

    batch_size = 100
    # drop_out_ratio = (训练时)0.8/(测试时)1
    with tf.Session() as sess:
        sess.run(init)
        # save.restore(sess, save_path=save_path)
        # plt.ion()
        writer = tf.summary.FileWriter("./CNN_Net_Tensorflow_logs", sess.graph)
        loss_list = []
        train_acc_list = []
        test_acc_list = []
        epoch_list = []

        for epoch in range(1000):
            train_x, train_y = mnist.train.next_batch(batch_size)
            train_in_x = train_x.reshape([batch_size, 28, 28, 1])

            summary, _, loss, train_acc = sess.run([merged, cnn.opt, cnn.loss, cnn.accu],
                                                   feed_dict={cnn.x: train_in_x, cnn.y: train_y, cnn.dp: 0.8})
            writer.add_summary(summary, epoch)

            if epoch % 100 == 0:
                test_x, test_y = mnist.test.next_batch(batch_size)
                test_in_x = test_x.reshape([batch_size, 28, 28, 1])
                test_output, test_acc = sess.run([cnn.output, cnn.accu],
                                                 feed_dict={cnn.x: test_in_x, cnn.y: test_y, cnn.dp: 1})
                loss_list.append(loss)
                # print(loss_list)
                train_acc_list.append(train_acc)
                # print(train_acc_list)
                test_acc_list.append(test_acc)
                # print(test_acc_list)
                epoch_list.append(epoch)
                print('--------------------------------------------------------------------')
                print('test_output:', np.argmax(test_output, 1)[:10])
                print('test_label:', np.argmax(test_y, 1)[:10])
                print('epoch:{}, loss:{:.4f}, train_acc:{:.2f}%, test_acc:{:.2f}%'.format(epoch, loss, train_acc * 100,
                                                                                          test_acc * 100))
                print('--------------------------------------------------------------------')
                save.save(sess, save_path=save_path)
        plt.plot(epoch_list, loss_list)
        plt.plot(epoch_list, train_acc_list)
        plt.plot(epoch_list, test_acc_list)
        plt.show()
        plt.pause(1)

        # plt.ioff()
