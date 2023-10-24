
# -*- coding: utf-8 -*-
import os
import math
import numpy as np
import tensorflow as tf
import openpyxl
#import matplotlib.pyplot as plt

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
# ============================================================================
# -----------------生成图片路径和标签的List------------------------------------

wb = openpyxl.workbook.Workbook()
sheet = wb.active

train_dir = './dataset/image'


Loadup = []
label_Loadup = []
UnLoadup = []
label_UnLoadup = []

# step1：获取'下所有的图片路径名，存放到
# 对应的列表中，同时贴上标签，存放到label列表中。
def get_files(file_dir, ratio):
    for file in os.listdir(file_dir + '/U'):
        UnLoadup.append(file_dir + '/U' + '/' + file)
        label_UnLoadup.append(0)
    for file in os.listdir(file_dir + '/L'):
        Loadup.append(file_dir + '/L' + '/' + file)
        label_Loadup.append(1)

    # step2：对生成的图片路径和标签List做打乱处理把cat和dog合起来组成一个list（img和lab）
    image_list = np.hstack((Loadup, UnLoadup))
    label_list = np.hstack((label_Loadup, label_UnLoadup))

    # 利用shuffle打乱顺序
    temp = np.array([image_list, label_list])
    temp = temp.transpose()
    np.random.shuffle(temp)

    # 从打乱的temp中再取出list（img和lab）
    # image_list = list(temp[:, 0])
    # label_list = list(temp[:, 1])
    # label_list = [int(i) for i in label_list]
    # return image_list, label_list

    # 将所有的img和lab转换成list
    all_image_list = list(temp[:, 0])
    all_label_list = list(temp[:, 1])

    # 将所得List分为两部分，一部分用来训练tra，一部分用来测试val
    # ratio是测试集的比例
    n_sample = len(all_label_list)
    n_val = int(math.ceil(n_sample * ratio))  # 测试样本数
    n_train = n_sample - n_val  # 训练样本数

    tra_images = all_image_list[0:n_train]
    tra_labels = all_label_list[0:n_train]
    tra_labels = [int(float(i)) for i in tra_labels]
    val_images = all_image_list[n_train:-1]
    val_labels = all_label_list[n_train:-1]
    val_labels = [int(float(i)) for i in val_labels]

    return tra_images, tra_labels, val_images, val_labels


# ---------------------------------------------------------------------------
# --------------------生成Batch----------------------------------------------

# step1：将上面生成的List传入get_batch() ，转换类型，产生一个输入队列queue，因为img和lab
# 是分开的，所以使用tf.train.slice_input_producer()，然后用tf.read_file()从队列中读取图像
#   image_W, image_H, ：设置好固定的图像高度和宽度
#   设置batch_size：每个batch要放多少张图片
#   capacity：一个队列最大多少
def get_batch(image, label, image_W, image_H, batch_size, capacity):
    # 转换类型
    image = tf.cast(image, tf.string)
    label = tf.cast(label, tf.int32)
    print(image)
    # make an input queue
    input_queue = tf.train.slice_input_producer([image, label])

    label = input_queue[1]
    image_contents = tf.read_file(input_queue[0])  # read img from a queue

    # step2：将图像解码，不同类型的图像不能混在一起，要么只用jpeg，要么只用png等。
    image = tf.image.decode_jpeg(image_contents, channels=3)

    # step3：数据预处理，对图像进行旋转、缩放、裁剪、归一化等操作，让计算出的模型更健壮。
    # image = tf.image.resize_image_with_crop_or_pad(image, image_W, image_H)
    # 灰度处理
    image = tf.image.rgb_to_grayscale(image)
    # 缩放操作 image_W , image_H 新图片的长和高
    image = tf.image.resize_images(image,[image_W, image_H])

    image = tf.image.per_image_standardization(image)

    # step4：生成batch
    # image_batch: 4D tensor [batch_size, width, height, 3],dtype=tf.float32
    # label_batch: 1D tensor [batch_size], dtype=tf.int32
    image_batch, label_batch = tf.train.batch([image, label],
                                              batch_size=batch_size,
                                              num_threads=32,
                                              capacity=capacity)
    # 重新排列label，行数为[batch_size]
    label_batch = tf.reshape(label_batch, [batch_size])
    image_batch = tf.cast(image_batch, tf.float32)
    return image_batch, label_batch

#
# epoch = 1
# batch_size_ = 20


def one_hot(labels, Label_class):
    one_hot_label = np.array([[int(i == int(labels[j])) for i in range(Label_class)] for j in range(len(labels))])
    return one_hot_label


# initial weights
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.02)
    return tf.Variable(initial)


# initial bias
def bias_variable(shape):
    initial = tf.constant(0.0, shape=shape)
    return tf.Variable(initial)


# convolution layer
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def conv4d(x, W):
    return tf.nn.conv4d(x, W, strides=[1, 1, 1, 1], padding='SAME')


# max_pool layer
def max_pool_4x4(x):
    return tf.nn.max_pool(x, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding='SAME')


def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


def train(train_dir,image_W,image_H,epoch=20,batch_size_=50):
    with tf.name_scope('Input'):
        x  = tf.placeholder(tf.float32, [batch_size_,image_W,image_H,1])
        y_ = tf.placeholder(tf.float32, [batch_size_, 2])

    with tf.name_scope('Conv_1'):
        # first convolution and max_pool layer
        W_conv1 = weight_variable([4, 4, 1, 16])
        b_conv1 = bias_variable([16])
        h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)
        # h_pool1 = max_pool_4x4(h_conv1)
    with tf.name_scope('Pool_1'):
        h_pool1 = max_pool_4x4(h_conv1)
    with tf.name_scope('Conv_2'):
        # second convolution and max_pool layer
        W_conv2 = weight_variable([5, 5, 16, 32])
        b_conv2 = bias_variable([32])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    with tf.name_scope('Pool_2'):
        h_pool2 = max_pool_4x4(h_conv2)  # 2*2

    # W_conv3 = weight_variable([5, 5, 64, 128])
    # b_conv3 = bias_variable([128])
    # h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
    # h_pool3 = max_pool_4x4(h_conv3)  # 2*2
    with tf.name_scope('FCL1'):
    # 变成全连接层，用一个MLP处理
        reshape = tf.reshape(h_pool2, [batch_size_, -1])
        dim = reshape.get_shape()[1].value
        W_fc1 = weight_variable([dim, 1024])
        b_fc1 = bias_variable([1024])
        h_fc1 = tf.nn.relu(tf.matmul(reshape, W_fc1) + b_fc1)

    #     variable_summaries(W_fc1)
    #     variable_summaries(b_fc1)
    #     variable_summaries(h_fc1)
    keep_prob = tf.placeholder(tf.float32, None)
    with tf.name_scope('FCL2_Dropout'):
    # dropout
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
        W_fc2 = weight_variable([1024,1024])
        b_fc2 = bias_variable([1024])
        h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop,W_fc2)+b_fc2)
        h_fc2_drop  = tf.nn.dropout(h_fc2, keep_prob)
    with tf.name_scope('FCL3_Dropout'):
        W_fc3 = weight_variable([1024, 512])
        b_fc3 = bias_variable([512])
        h_fc3 = tf.nn.relu(tf.matmul(h_fc2_drop, W_fc3) + b_fc3)
        h_fc3_drop = tf.nn.dropout(h_fc3, keep_prob)
    with tf.name_scope('FCL4_Dropout'):
        W_fc4 = weight_variable([512, 256])
        b_fc4 = bias_variable([256])
        h_fc4 = tf.nn.relu(tf.matmul(h_fc3_drop, W_fc4) + b_fc4)
        h_fc4_drop = tf.nn.dropout(h_fc4, keep_prob)
    with tf.name_scope('Output'):
        W_fc5 = weight_variable([256, 2])
        b_fc5 = bias_variable([2])
        y_conv = tf.nn.softmax(tf.matmul(h_fc4_drop, W_fc5) + b_fc5)

    # predicted = tf.argmax(y_conv,1)
    #     variable_summaries(W_fc2)
    #     variable_summaries(b_fc2)
    #     variable_summaries(y_conv)

    # loss = tf.reduce_mean(tf.square(y-prediction))
    # cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
    # 使用梯度下降法
    # train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)
    with tf.name_scope('cross_entropy'):
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
        tf.summary.scalar("cross_entropy", cross_entropy)

    # train_step =tf.train.AdadeltaOptimizer(learning_rate=0.9).minimize(cross_entropy)
    # train_step = tf.train.GradientDescentOptimizer(0.001).minimize(cross_entropy)
    with tf.name_scope('accuracy'):
        correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar("accuracy", accuracy)
    with tf.name_scope('train'):
        train_step = tf.train.AdamOptimizer(0.00005).minimize(cross_entropy)
    tra_images, tra_labels, val_images, val_labels = get_files(train_dir, 0.3)
    # print(len(tra_images),len(val_images))
    img_batchs, label_batchs = get_batch(tra_images, tra_labels, image_W,image_H, batch_size_, 300)
    img_val_batchs, label_val_batchs = get_batch(val_images, val_labels, image_W,image_H, batch_size_, 300)

    init = tf.initialize_all_variables()
    t_vars = tf.trainable_variables()
    print(t_vars)
    # merged = tf.merge_all_summaries()

    # train_writer = tf.train.SummaryWriter(  '/train',sess.graph)

    saver = tf.train.Saver()
    saver_path = 'save_7_26_3/model.ckpt'
    with tf.Session(config=tf.ConfigProto(device_count={"GPU":1 })) as sess:
        sess.run(init)
        global sheet
        merged = tf.summary.merge_all()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)
        batch_idxs = int(len(tra_images) / batch_size_)
        train_writer = tf.summary.FileWriter('F:\Desktop\TF_CNN\logs_726', sess.graph)
        n = 0
        for i in range(1,epoch):

            for j in range(batch_idxs):
                n += 1
                # run_metadata = tf.RunMetadata()
                val, l = sess.run([img_batchs, label_batchs])
                l = one_hot(l, 2)
                _, summary,acc ,cross= sess.run([train_step,merged, accuracy,cross_entropy], feed_dict={x: val, y_: l, keep_prob: 0.6})
                                                # run_metadata=run_metadata)
                # train_writer.add_run_metadata(run_metadata, '%d' % n)
                if n %10 ==0:
                    train_writer.add_summary(summary, '%d' % n)
                print("Epoch:[%4d] [%4d/%4d], accuracy:[%.8f]" % (i, j, batch_idxs, acc),"cross_entropy:",(cross))
            # sheet['A%s' % n] = n
            # sheet['B%s' % n] = acc
            # sheet['C%s' % n] = cross

        train_writer.add_graph(sess.graph)
        # wb.save('tf1.csv')
        # print(l)
        if os.path.isdir(saver_path.split('/')[0]):
            save_path = saver.save(sess, saver_path)
            print('save')
        else:
            os.makedirs(saver_path.split('/')[0])
            save_path = saver.save(sess, saver_path)
            print('save')
        batch_idval = int(len(val_images) / batch_size_)
        for test_i in range(batch_idval):
            val, l = sess.run([img_val_batchs, label_val_batchs])
            l = one_hot(l, 2)
            y, acc = sess.run([y_conv, accuracy], feed_dict={x: val, y_: l, keep_prob: 1.0})
            # print(y)
            # print(sess.run(tf.argmax(y, 1)))
            print("test accuracy: [%.8f]" % (acc))

        train_writer.close()
        coord.request_stop()
        coord.join(threads)

if __name__ == '__main__':


    train(train_dir, 150, 100,150)







