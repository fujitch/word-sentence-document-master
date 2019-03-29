# -*- coding: utf-8 -*-

import pickle
import tensorflow as tf
import numpy as np
import MeCab
from gensim.models import word2vec

m_owakati = MeCab.Tagger(r'-Owakati -d C:\Users\hori\workspace\encoder-decoder-sentence-chainer-master\mecab-ipadic-neologd')
title_dict = pickle.load(open("energy_paper_2018_title.pickle", "rb"))
model = pickle.load(open('word2vec_neologd30.pickle', 'rb'))
length_title_dict = len(title_dict)
dict_keys_list = list(title_dict.keys())
max_word_size = 186

def make_batch(key):
    batch_size = len(title_dict[key])
    batch = np.zeros((batch_size, max_word_size * 200))
    batch = np.array(batch, dtype=np.float32)
    output = np.zeros((batch_size, length_title_dict))
    output = np.array(output, dtype=np.int32)
    datas = title_dict[key]
    for ind in range(len(datas)):
        data = datas[ind]
        data = data[data.find("】") + 1:]
        data = m_owakati.parse(data).split(" ")
        data = data[:len(data)-1]
        batch_dum = np.zeros((max_word_size, 200))
        batch_dum = np.array(batch_dum, dtype=np.float32)
        for i in range(len(data)):
            d = data[i]
            try:
                batch_dum[i, :] = model.wv[d]
            except:
                continue
        batch[ind, :] = np.reshape(batch_dum, (max_word_size * 200))
        output[ind, dict_keys_list.index(key)] = 1
    return batch, output

tf.reset_default_graph()

x = tf.placeholder("float", shape=[None, max_word_size * 200])
y_ = tf.placeholder("float", shape=[None, len(title_dict)])

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
saver.restore(sess, "./model20190204/train-sentence")
sort_dict = {}
out_list = []


k = '2-1'
batch, output = make_batch(k)
out_titles = y_out.eval(session=sess, feed_dict={x: batch, y_: output, keep_prob: 1.0})
"""
for k in dict_keys_list:
    batch, output = make_batch(k)
    out_titles = y_out.eval(session=sess, feed_dict={x: batch, y_: output, keep_prob: 1.0})
    out_list.append(out_titles)
""" 
    