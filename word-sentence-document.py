# -*- coding: utf-8 -*-

import pickle
import tensorflow as tf
from tensorflow.contrib.framework import sort
import numpy as np
import random

part_chapter_dict_wakati = pickle.load(open("part_chapter_dict_wakati.pickle", "rb"))
length_part_chapter_dict_wakati = len(part_chapter_dict_wakati)
batch_size = length_part_chapter_dict_wakati
training_epochs = 10000
display_epochs = 100

max_sentence_size = 0
max_word_size = 0

for k in part_chapter_dict_wakati:
    for sentence_list in part_chapter_dict_wakati[k]:
        if max_sentence_size < len(sentence_list):
            max_sentence_size = len(sentence_list)
        for sentence in sentence_list:
            if max_word_size < sentence.shape[0]:
                max_word_size = sentence.shape[0]

dict_keys_list = list(part_chapter_dict_wakati.keys())

all_max = 0.0
all_min = 0.0
for k in part_chapter_dict_wakati:
    for sentence_list in part_chapter_dict_wakati[k]:
        for sentence in sentence_list:
            if sentence.shape[0] == 0:
                continue
            if all_max < np.max(sentence):
                all_max = np.max(sentence)
            if all_min > np.min(sentence):
                all_min = np.min(sentence)

dum_part_chapter_dict_wakati = {}
for k in part_chapter_dict_wakati:
    dum_sentence_list = []
    for sentence_list in part_chapter_dict_wakati[k]:
        dum_sentence = []
        for sentence in sentence_list:
            dum_sentence.append((sentence - all_min) / (all_max - all_min))
        dum_sentence_list.append(dum_sentence)
    dum_part_chapter_dict_wakati[k] = dum_sentence_list

part_chapter_dict_wakati = dum_part_chapter_dict_wakati

def make_batch():
    batch = np.zeros((batch_size, max_sentence_size * max_word_size * 200))
    batch = np.array(batch, dtype=np.float32)
    output = np.zeros((batch_size, length_part_chapter_dict_wakati))
    output = np.array(output, dtype=np.int32)
    for i in range(batch_size):
        datas = part_chapter_dict_wakati[dict_keys_list[i%length_part_chapter_dict_wakati]][random.randint(0, len(part_chapter_dict_wakati[dict_keys_list[i%length_part_chapter_dict_wakati]])-1)]
        batch_before = np.zeros((max_word_size, 200, max_sentence_size))
        for k in range(len(datas)):
            data = datas[k]
            dum_matrix = np.zeros((max_word_size, 200))
            dum_matrix[:data.shape[0], :] = data
            batch_before[:, :, k] = dum_matrix
        batch[i, :] = np.reshape(batch_before, (max_sentence_size * max_word_size * 200))
        output[i, i%length_part_chapter_dict_wakati] = 1
    return batch, output

tf.reset_default_graph()

x = tf.placeholder("float", shape=[None, max_sentence_size * max_word_size * 200])
y_ = tf.placeholder("float", shape=[None, len(part_chapter_dict_wakati)])

# 荷重作成
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

# バイアス作成
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# 畳み込み処理を定義
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='VALID')

# k-maxプーリング処理を定義
def k_max_pool_4(x):
    return sort(tf.nn.top_k(x, k=4)[1])

# 畳み込み層1
doc_matrix = tf.zeros([batch_size, max_sentence_size, 256 * 4, 1])
doc_matrix = tf.Variable(tf.cast(doc_matrix, tf.float32))

W_conv1 = weight_variable([3, 200, 1, 256])
b_conv1 = bias_variable([256])
x_image = tf.reshape(x, [-1, max_word_size, 200, max_sentence_size])

x_image0 = tf.reshape(x_image[:, :, :, 0], [-1, max_word_size, 200, 1])
h_conv1_0 = tf.nn.relu(conv2d(x_image0, W_conv1) + b_conv1)
h_conv1_0_trans = tf.transpose(h_conv1_0, [0, 2, 3, 1])

x_image1 = tf.reshape(x_image[:, :, :, 1], [-1, max_word_size, 200, 1])
h_conv1_1 = tf.nn.relu(conv2d(x_image1, W_conv1) + b_conv1)
h_conv1_1_trans = tf.transpose(h_conv1_1, [0, 2, 3, 1])

x_image2 = tf.reshape(x_image[:, :, :, 2], [-1, max_word_size, 200, 1])
h_conv1_2 = tf.nn.relu(conv2d(x_image2, W_conv1) + b_conv1)
h_conv1_2_trans = tf.transpose(h_conv1_2, [0, 2, 3, 1])

x_image3 = tf.reshape(x_image[:, :, :, 3], [-1, max_word_size, 200, 1])
h_conv1_3 = tf.nn.relu(conv2d(x_image3, W_conv1) + b_conv1)
h_conv1_3_trans = tf.transpose(h_conv1_3, [0, 2, 3, 1])

x_image4 = tf.reshape(x_image[:, :, :, 4], [-1, max_word_size, 200, 1])
h_conv1_4 = tf.nn.relu(conv2d(x_image4, W_conv1) + b_conv1)
h_conv1_4_trans = tf.transpose(h_conv1_4, [0, 2, 3, 1])

x_image5 = tf.reshape(x_image[:, :, :, 5], [-1, max_word_size, 200, 1])
h_conv1_5 = tf.nn.relu(conv2d(x_image5, W_conv1) + b_conv1)
h_conv1_5_trans = tf.transpose(h_conv1_5, [0, 2, 3, 1])

x_image6 = tf.reshape(x_image[:, :, :, 6], [-1, max_word_size, 200, 1])
h_conv1_6 = tf.nn.relu(conv2d(x_image6, W_conv1) + b_conv1)
h_conv1_6_trans = tf.transpose(h_conv1_6, [0, 2, 3, 1])

x_image7 = tf.reshape(x_image[:, :, :, 7], [-1, max_word_size, 200, 1])
h_conv1_7 = tf.nn.relu(conv2d(x_image7, W_conv1) + b_conv1)
h_conv1_7_trans = tf.transpose(h_conv1_7, [0, 2, 3, 1])

x_image8 = tf.reshape(x_image[:, :, :, 8], [-1, max_word_size, 200, 1])
h_conv1_8 = tf.nn.relu(conv2d(x_image8, W_conv1) + b_conv1)
h_conv1_8_trans = tf.transpose(h_conv1_8, [0, 2, 3, 1])

x_image9 = tf.reshape(x_image[:, :, :, 9], [-1, max_word_size, 200, 1])
h_conv1_9 = tf.nn.relu(conv2d(x_image9, W_conv1) + b_conv1)
h_conv1_9_trans = tf.transpose(h_conv1_9, [0, 2, 3, 1])

x_image10 = tf.reshape(x_image[:, :, :, 10], [-1, max_word_size, 200, 1])
h_conv1_10 = tf.nn.relu(conv2d(x_image10, W_conv1) + b_conv1)
h_conv1_10_trans = tf.transpose(h_conv1_10, [0, 2, 3, 1])

x_image11 = tf.reshape(x_image[:, :, :, 11], [-1, max_word_size, 200, 1])
h_conv1_11 = tf.nn.relu(conv2d(x_image11, W_conv1) + b_conv1)
h_conv1_11_trans = tf.transpose(h_conv1_11, [0, 2, 3, 1])

x_image12 = tf.reshape(x_image[:, :, :, 12], [-1, max_word_size, 200, 1])
h_conv1_12 = tf.nn.relu(conv2d(x_image12, W_conv1) + b_conv1)
h_conv1_12_trans = tf.transpose(h_conv1_12, [0, 2, 3, 1])

x_image13 = tf.reshape(x_image[:, :, :, 13], [-1, max_word_size, 200, 1])
h_conv1_13 = tf.nn.relu(conv2d(x_image13, W_conv1) + b_conv1)
h_conv1_13_trans = tf.transpose(h_conv1_13, [0, 2, 3, 1])

x_image14 = tf.reshape(x_image[:, :, :, 14], [-1, max_word_size, 200, 1])
h_conv1_14 = tf.nn.relu(conv2d(x_image14, W_conv1) + b_conv1)
h_conv1_14_trans = tf.transpose(h_conv1_14, [0, 2, 3, 1])

x_image15 = tf.reshape(x_image[:, :, :, 15], [-1, max_word_size, 200, 1])
h_conv1_15 = tf.nn.relu(conv2d(x_image15, W_conv1) + b_conv1)
h_conv1_15_trans = tf.transpose(h_conv1_15, [0, 2, 3, 1])

x_image16 = tf.reshape(x_image[:, :, :, 16], [-1, max_word_size, 200, 1])
h_conv1_16 = tf.nn.relu(conv2d(x_image16, W_conv1) + b_conv1)
h_conv1_16_trans = tf.transpose(h_conv1_16, [0, 2, 3, 1])

# プーリング層1
doc_matrix[:, 0, :, 0].assign(tf.reshape(tf.cast(k_max_pool_4(h_conv1_0_trans), tf.float32), [-1, 256 * 4]))
doc_matrix[:, 1, :, 0].assign(tf.reshape(tf.cast(k_max_pool_4(h_conv1_1_trans), tf.float32), [-1, 256 * 4]))
doc_matrix[:, 2, :, 0].assign(tf.reshape(tf.cast(k_max_pool_4(h_conv1_2_trans), tf.float32), [-1, 256 * 4]))
doc_matrix[:, 3, :, 0].assign(tf.reshape(tf.cast(k_max_pool_4(h_conv1_3_trans), tf.float32), [-1, 256 * 4]))
doc_matrix[:, 4, :, 0].assign(tf.reshape(tf.cast(k_max_pool_4(h_conv1_4_trans), tf.float32), [-1, 256 * 4]))
doc_matrix[:, 5, :, 0].assign(tf.reshape(tf.cast(k_max_pool_4(h_conv1_5_trans), tf.float32), [-1, 256 * 4]))
doc_matrix[:, 6, :, 0].assign(tf.reshape(tf.cast(k_max_pool_4(h_conv1_6_trans), tf.float32), [-1, 256 * 4]))
doc_matrix[:, 7, :, 0].assign(tf.reshape(tf.cast(k_max_pool_4(h_conv1_7_trans), tf.float32), [-1, 256 * 4]))
doc_matrix[:, 8, :, 0].assign(tf.reshape(tf.cast(k_max_pool_4(h_conv1_8_trans), tf.float32), [-1, 256 * 4]))
doc_matrix[:, 9, :, 0].assign(tf.reshape(tf.cast(k_max_pool_4(h_conv1_9_trans), tf.float32), [-1, 256 * 4]))
doc_matrix[:, 10, :, 0].assign(tf.reshape(tf.cast(k_max_pool_4(h_conv1_10_trans), tf.float32), [-1, 256 * 4]))
doc_matrix[:, 11, :, 0].assign(tf.reshape(tf.cast(k_max_pool_4(h_conv1_11_trans), tf.float32), [-1, 256 * 4]))
doc_matrix[:, 12, :, 0].assign(tf.reshape(tf.cast(k_max_pool_4(h_conv1_12_trans), tf.float32), [-1, 256 * 4]))
doc_matrix[:, 13, :, 0].assign(tf.reshape(tf.cast(k_max_pool_4(h_conv1_13_trans), tf.float32), [-1, 256 * 4]))
doc_matrix[:, 14, :, 0].assign(tf.reshape(tf.cast(k_max_pool_4(h_conv1_14_trans), tf.float32), [-1, 256 * 4]))
doc_matrix[:, 15, :, 0].assign(tf.reshape(tf.cast(k_max_pool_4(h_conv1_15_trans), tf.float32), [-1, 256 * 4]))
doc_matrix[:, 16, :, 0].assign(tf.reshape(tf.cast(k_max_pool_4(h_conv1_16_trans), tf.float32), [-1, 256 * 4]))

# 畳み込み層2
W_conv2 = weight_variable([3, 256 * 4, 1, 256])
b_conv2 = bias_variable([256])
h_conv2 = tf.nn.relu(conv2d(doc_matrix, W_conv2) + b_conv2)
h_conv2_trans = tf.transpose(h_conv2, [0, 2, 3, 1])
h_pool2 = tf.cast(k_max_pool_4(h_conv2_trans), tf.float32)

# 全結合層1
W_fc1 = weight_variable([256 * 4, 128])
b_fc1 = bias_variable([128])
h_flat = tf.reshape(h_pool2, [-1, 256 * 4])
h_fc1 = tf.nn.relu(tf.matmul(h_flat, W_fc1) + b_fc1)

# 全結合層2
W_fc2 = weight_variable([128, 14])
b_fc2 = bias_variable([14])
y_out = tf.nn.softmax(tf.matmul(h_fc1, W_fc2) + b_fc2)

cross_entropy = -tf.reduce_sum(y_ * tf.log(tf.clip_by_value(y_out,1e-10,1.0)))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_out, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

sess = tf.Session()
saver = tf.train.Saver()
sess.run(tf.initialize_all_variables())

for i in range(training_epochs):
    batch, output = make_batch()
    train_accuracy = accuracy.eval(session=sess, feed_dict={x: batch, y_: output})
    out = y_out.eval(session=sess, feed_dict={x: batch, y_: output})
    print(out)
    if i%display_epochs ==0:
        loss = cross_entropy.eval(session=sess, feed_dict={x: batch, y_: output})
        print(i)
        print(train_accuracy)
        print(loss)
    train_step.run(session=sess, feed_dict={x: batch, y_: output})