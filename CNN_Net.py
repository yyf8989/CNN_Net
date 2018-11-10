import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
import numpy as np


class CNNnet:
    def __init__(self):
        self.x = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28, 1])
        self.y = tf.placeholder(dtype=tf.float32, shape=[None, 10])

        self.dp = tf.placeholder(tf.float32)

        self.conv1_w = tf.Variable(tf.truncated_normal(shape=[3, 3, 1, 16], stddev=0.02))
        self.conv1_b = tf.Variable(tf.zeros([16]))

        self.conv2_w = tf.Variable(tf.truncated_normal(shape=[3, 3, 16, 32], stddev=0.02))
        self.conv2_b = tf.Variable(tf.zeros([32]))


    def forward(self):

        self.conv1 = tf.nn.relu(tf.nn.conv2d(self.x, self.conv1_w, strides=[1, 1, 1, 1], padding='SAME',
                                             use_cudnn_on_gpu=True) + self.conv1_b)
        self.pooling1 = tf.nn.max_pool(self.conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        self.conv2 = tf.nn.relu(tf.nn.conv2d(self.pooling1, self.conv2_w, strides=[1, 1, 1, 1], padding='SAME',
                                             use_cudnn_on_gpu=True) + self.conv2_b)
        self.pooling2 = tf.nn.max_pool(self.conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

        self.flat = tf.reshape(self.pooling2, [-1, 7*7*32])

        self.fc1_w = tf.Variable(tf.truncated_normal([7*7*32, 128], dtype=tf.float32, stddev=0.02))
        self.fc1_b = tf.Variable(tf.zeros([128]))

        self.fc1 = tf.nn.relu(tf.matmul(self.flat, self.fc1_w) + self.fc1_b)

        self.fc2_w = tf.Variable(tf.truncated_normal([128, 10], dtype=tf.float32, stddev=0.02))
        self.fc2_b = tf.Variable(tf.zeros([10]))

        self.drop = tf.nn.dropout(self.fc1, keep_prob=self.dp)

        self.out_put = tf.nn.softmax(tf.matmul(self.drop, self.fc2_w) + self.fc2_b)



    def backward(self):

        self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.out_put, labels=self.y))

        # self.loss = tf.reduce_mean((self.out_put - self.y)**2)
        # self.opt = tf.train.AdamOptimizer(0.0001).minimize(self.loss)

        self.opt = tf.train.AdamOptimizer(0.001).minimize(self.cross_entropy)

        self.correct_prediction = tf.equal(tf.argmax(self.out_put, 1), tf.argmax(self.y, 1))
        self.rst = tf.cast(self.correct_prediction,'float')
        self.accuracy = tf.reduce_mean(self.rst)


if __name__ == '__main__':
    cnn_net = CNNnet()
    cnn_net.forward()
    cnn_net.backward()
    init = tf.global_variables_initializer()

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)


        # mx = np.array([])
        # my = np.array([])
        #
        # plt.axis([0,100,0,1])
        # plt.ion()

        for epoch in range(10001):
            train_x, train_y = mnist.train.next_batch(100)
            # print(train_x.shape)
            train_x = train_x.reshape([100, 28, 28, 1])
            # print(train_x.shape)
            _, _loss, acc = sess.run([cnn_net.opt, cnn_net.cross_entropy, cnn_net.accuracy ],
                                     feed_dict={cnn_net.x:train_x, cnn_net.y:train_y, cnn_net.dp:0.8})

            if epoch % 500 == 0:
                print('epoch:{0},精度:{1}'.format(epoch, acc))


