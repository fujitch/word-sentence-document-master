# -*- coding: utf-8 -*-

import pickle
import tensorflow as tf
import numpy as np
import random
import time

part_chapter_dict_wakati = pickle.load(open("part_chapter_dict_wakati.pickle", "rb"))
length_part_chapter_dict_wakati = len(part_chapter_dict_wakati)
batch_size = 100

training_epochs = 100000
display_epochs = 100

max_word_size = 0

for k in part_chapter_dict_wakati:
    for sentence_list in part_chapter_dict_wakati[k]:
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
    batch = np.zeros((batch_size, max_word_size * 200))
    batch = np.array(batch, dtype=np.float32)
    output = np.zeros((batch_size, length_part_chapter_dict_wakati))
    output = np.array(output, dtype=np.int32)
    for i in range(batch_size):
        datas = part_chapter_dict_wakati[dict_keys_list[i%length_part_chapter_dict_wakati]][random.randint(0, len(part_chapter_dict_wakati[dict_keys_list[i%length_part_chapter_dict_wakati]])-1)]
        data = datas[random.randint(0, len(datas)-1)]
        batch_before = np.zeros((max_word_size, 200))
        dum_matrix = np.zeros((max_word_size, 200))
        dum_matrix[:data.shape[0], :] = data
        batch_before[:, :] = dum_matrix
        batch[i, :] = np.reshape(batch_before, (max_word_size * 200))
        output[i, i%length_part_chapter_dict_wakati] = 1
    return batch, output

tf.reset_default_graph()

x = tf.placeholder("float", shape=[None, max_word_size * 200])
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

# プーリング処理を定義
def max_pool_2_2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
    
W_conv1 = weight_variable([3, 3, 1, 64])
b_conv1 = bias_variable([64])

W_conv2 = weight_variable([3, 3, 64, 64])
b_conv2 = bias_variable([64])

W_conv3 = weight_variable([4, 4, 64, 128])
b_conv3 = bias_variable([128])

W_conv4 = weight_variable([4, 4, 128, 128])
b_conv4 = bias_variable([128])

W_conv5 = weight_variable([4, 4, 128, 256])
b_conv5 = bias_variable([256])

W_fc1 = weight_variable([3 * 4 * 256, 128])
b_fc1 = bias_variable([128])

W_fc2 = weight_variable([128, 14])
b_fc2 = bias_variable([14])

x_image = tf.reshape(x, [-1, max_word_size, 200])

x_image = tf.reshape(x_image, [-1, max_word_size, 200, 1])
h_conv1 = max_pool_2_2(tf.nn.relu(conv2d(max_pool_2_2(tf.nn.relu(conv2d(max_pool_2_2(tf.nn.relu(conv2d(max_pool_2_2(tf.nn.relu(conv2d(max_pool_2_2(tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)), W_conv2) + b_conv2)), W_conv3) + b_conv3)), W_conv4) + b_conv4)), W_conv5) + b_conv5))
h_flat1 = tf.reshape(h_conv1, [-1, 3 * 4 * 256])
h_fc1 = tf.nn.relu(tf.matmul(h_flat1, W_fc1) + b_fc1)

# ドロップアウト
keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

y_out = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

cross_entropy = -tf.reduce_sum(y_ * tf.log(tf.clip_by_value(y_out,1e-10,1.0)))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_out, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

sess = tf.Session()
saver = tf.train.Saver()
sess.run(tf.initialize_all_variables())

for i in range(training_epochs):
    batch, output = make_batch()
    batch_zeros = np.zeros((batch_size, max_word_size * 200))
    batch_zeros = np.array(batch_zeros, dtype=np.float32)
    if i%display_epochs == 0:
        train_accuracy = accuracy.eval(session=sess, feed_dict={x: batch, y_: output, keep_prob: 1.0})
        loss = cross_entropy.eval(session=sess, feed_dict={x: batch, y_: output, keep_prob: 1.0})
        print(i)
        print(train_accuracy)
        print(loss)
    train_step.run(session=sess, feed_dict={x: batch, y_: output, keep_prob: 0.5})
saver.save(sess, "./model20190204/train-sentence")
sess.close()