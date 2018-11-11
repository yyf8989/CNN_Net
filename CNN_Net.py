import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data

mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
import numpy as np

save_path = r"E:\Pycharmprojects\CNN_Net\CNN_Net_ckpt\1_ckpt"


class CNNnet:
    # 定义CNN网络类
    def __init__(self):
        self.x = tf.placeholder(dtype=tf.float32, shape=[None, 28, 28, 1])
        # 定义输入层placeholder,输入为(NHWC)
        self.y = tf.placeholder(dtype=tf.float32, shape=[None, 10])
        # 定义输出层placeholder

        self.dp = tf.placeholder(tf.float32)
        # 定义drpout率的placeholder

        self.conv1_w = tf.Variable(tf.truncated_normal(shape=[3, 3, 1, 16], stddev=0.02))
        # 定义卷积核1相关参数[3,3,1,16（卷积核个数）],使用截断正态分布
        self.conv1_b = tf.Variable(tf.zeros([16]))
        # 定义偏值b，可利用广播进行维度变换，后续均类似

        self.conv2_w = tf.Variable(tf.truncated_normal(shape=[3, 3, 16, 32], stddev=0.02))
        # 定义卷积核2相关参数[3,3,16,32（卷积核个数）],使用截断正态分布
        self.conv2_b = tf.Variable(tf.zeros([32]))

    def forward(self):
        # 定义前向传播

        self.conv1 = tf.nn.relu(tf.nn.conv2d(self.x, self.conv1_w, strides=[1, 1, 1, 1], padding='SAME',
                                             use_cudnn_on_gpu=True) + self.conv1_b)
        # 第1个卷积层，步长为1，padding为SAME 输出形状为[batch_size, 28, 28, 16]

        self.pooling1 = tf.nn.max_pool(self.conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        # 第1个卷积层后跟着池化层，size=[2,2], 步长为2，padding为SAME,输出形状为[batch_size, 14, 14, 16]

        self.conv2 = tf.nn.relu(tf.nn.conv2d(self.pooling1, self.conv2_w, strides=[1, 1, 1, 1], padding='SAME',
                                             use_cudnn_on_gpu=True) + self.conv2_b)
        # 第2个卷积层，步长为1，padding为SAME 输出形状为[batch_size, 14, 14, 32]

        self.pooling2 = tf.nn.max_pool(self.conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        # 第2个卷积层后跟着池化层，size=[2,2], 步长为2，padding为SAME,输出形状为[batch_size, 7, 7, 32]

        self.flat = tf.reshape(self.pooling2, [-1, 7 * 7 * 32])
        # 将卷积结果输出结果由NHWC--> NV结构进行全连接输入

        self.fc1_w = tf.Variable(tf.truncated_normal([7 * 7 * 32, 128], dtype=tf.float32, stddev=0.02))
        # 定义全连接1层权重Wshape=[NV] --> [7*7*32, 128]
        self.fc1_b = tf.Variable(tf.zeros([128]))
        # 定义偏值b

        self.fc1 = tf.nn.relu(tf.matmul(self.flat, self.fc1_w) + self.fc1_b)
        # 进行全连接输出，输出后使用relu函数激活

        self.fc2_w = tf.Variable(tf.truncated_normal([128, 10], dtype=tf.float32, stddev=0.02))
        # 定义全连接2层权重Wshape=[NV] --> [128, 10]
        self.fc2_b = tf.Variable(tf.zeros([10]))
        # 定义偏值b

        self.drop = tf.nn.dropout(self.fc1, keep_prob=self.dp)
        # 全连接层1层之后之后先使用dropout进行输出

        self.out_put = tf.nn.softmax(tf.matmul(self.drop, self.fc2_w) + self.fc2_b)
        # dropout之后进行全连接2层输出，然后因为是10分类问题，故进行softmax激活输出

    def backward(self):
        self.cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.out_put, labels=self.y))
        # 使用softmax交叉熵计算损失，logits代表输出结果，labels代表标签

        # self.loss = tf.reduce_mean((self.out_put - self.y)**2)
        # self.opt = tf.train.AdamOptimizer(0.0001).minimize(self.loss)

        self.opt = tf.train.AdamOptimizer(0.0005).minimize(self.cross_entropy)
        # 使用Adam优化器，最小化交叉熵

        self.correct_prediction = tf.equal(tf.argmax(self.out_put, 1), tf.argmax(self.y, 1))
        # 将数据进行bool类型计算
        self.rst = tf.cast(self.correct_prediction, 'float')
        # 转换bool类型 --> 浮点数据类型
        self.accuracy = tf.reduce_mean(self.rst)
        # 进行准确率计算


if __name__ == '__main__':
    # 实例化相关方法
    cnn_net = CNNnet()
    cnn_net.forward()
    cnn_net.backward()
    init = tf.global_variables_initializer()
    # 初始化全部变量

    saver = tf.train.Saver()
    # 定义checkpointsave

    with tf.Session() as sess:
        sess.run(init)
        # saver.restore(sess, save_path=save_path)
        # 可restore checkpoint继续训练

        # mx = np.array([])
        # my = np.array([])
        #
        # plt.axis([0,100,0,1])
        # plt.ion()

        for epoch in range(10001):
            # 定义训练次数
            train_x, train_y = mnist.train.next_batch(100)
            # 去除mnist测试集中的数据
            # print(train_x.shape)
            train_x = train_x.reshape([100, 28, 28, 1])
            # 转换测试集数据shape NV--> NHWC
            # print(train_x.shape)
            _, _loss, acc = sess.run([cnn_net.opt, cnn_net.cross_entropy, cnn_net.accuracy],
                                     feed_dict={cnn_net.x: train_x, cnn_net.y: train_y, cnn_net.dp: 0.8})
            # 输入相关参数至placeholder，然后进行相关需要数据如loss，acc输出
            if epoch % 500 == 0:
                # 每500次打印一次精度
                print('epoch:{},精度:{:4.2f}%'.format(epoch, acc * 100))

                test_x, test_y = mnist.test.next_batch(100)
                test_x = test_x.reshape([100, 28, 28, 1])
                # 取测试集进行测试
                test_out = sess.run(cnn_net.out_put, feed_dict={cnn_net.x: test_x, cnn_net.y: test_y, cnn_net.dp: 1})
                # 不训练，只取用output的输出
                test_out_data = np.argmax(test_out[1])
                # 取出其中一张图片进行验证，只验证数字，没有打印出图片
                test_y_data = np.argmax(test_y[1])
                print("test_label:{}, test_output:{}".format(test_out_data, test_y_data))
                # 打印标签和训练出的数据
                saver.save(sess, save_path=save_path)
                # 保存checkpoint
