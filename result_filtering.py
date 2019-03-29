# -*- coding: utf-8 -*-

import MeCab
from gensim.models import word2vec
import pickle
import numpy as np
import tensorflow as tf
import random

m_owakati = MeCab.Tagger(r'-Owakati -d C:\Users\hori\workspace\encoder-decoder-sentence-chainer-master\mecab-ipadic-neologd')
m_ochasen = MeCab.Tagger(r'-Ochasen -d C:\Users\hori\workspace\encoder-decoder-sentence-chainer-master\mecab-ipadic-neologd')
model = pickle.load(open('word2vec_neologd30.pickle', 'rb'))
part_chapter_dict = pickle.load(open("energy_paper_2018_dict.pickle", "rb"))

part_chapter_dict_wakati = pickle.load(open("part_chapter_dict_wakati.pickle", "rb"))
length_part_chapter_dict_wakati = len(part_chapter_dict_wakati)
batch_size = length_part_chapter_dict_wakati

max_sentence_size = 0
max_word_size = 0

filter_figure_title = True
filter_part_title = True
filter_frequent_word = True
filter_connection_word = True
filter_position = True

figure_title = "最終エネルギー消費と実質GDPの推移"
part_title_list = ["エネルギー動向", "国内エネルギー動向", "エネルギー需給の概要", "エネルギー消費の動向"]
frequent_dict = pickle.load(open("frequent_dict.pickle", "rb"))



figure_title_word_list = []
node = m_ochasen.parseToNode(figure_title)
while node:
    fields = node.feature.split(",")
    if fields[0] == '名詞' or fields[0] == '動詞' or fields[0] == '形容詞':
        if node.surface not in figure_title_word_list:
            figure_title_word_list.append(node.surface)
    node = node.next

filter_part_title_list = []
for part_title in part_title_list:
    node = m_ochasen.parseToNode(part_title)
    while node:
        fields = node.feature.split(",")
        if fields[0] == '名詞' or fields[0] == '動詞' or fields[0] == '形容詞':
            if node.surface not in filter_part_title_list:
                filter_part_title_list.append(node.surface)
        node = node.next

connection_word_list = ["だから", "そのため", "このため", "したがって", "ゆえに", "それゆえに", "つまり", "すなわち", "要するに", "特に", "とりわけ", "中でも", "なかでも"]

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


texts = "1970年代までの高度経済成長期に、我が国のエネルギー消費は国内総生産（GDP）よりも高い伸び率で増加しました。しかし、1970年代の二度の石油ショックを契機に、製造業を中心に省エネルギー化が進むとともに、省エネルギー型製品の開発も盛んになりました。このような努力の結果、エネルギー消費を抑制しながら経済成長を果たすことができました。1990年代を通して原油価格が低水準で推移する中で、家庭部門、業務他部門を中心にエネルギー消費は増加しました。2000年代半ば以降は再び原油価格が上昇したこともあり、2004年度をピークに最終エネルギー消費は減少傾向になりました。2011年度からは東日本大震災以降の節電意識の高まりなどによってさらに減少が進みました。2016年度は実質GDPが2015年度より1.2%増加しましたが、前年度より省エネルギーが進展したことから、最終エネルギー消費は同1.3%減少しました（第211-1-1）。"
texts = texts.split("。")
if texts[len(texts)-1] == "":
    texts = texts[:len(texts)-1]
            
# 前処理
text_list = []
raw_text_list = []
for text in texts:
    raw_text_list.append(text)
    text_wakati_list = m_owakati.parse(text).split(" ")
    text_wakati_list = text_wakati_list[:len(text_wakati_list)-1]
    word_vector_matrix = np.zeros((len(text_wakati_list), 200))
    for i in range(len(text_wakati_list)):
        word = text_wakati_list[i]
        try:
            word_vector_matrix[i, :] = model.wv[word]
        except:
            continue
    word_vector_matrix -= all_min
    word_vector_matrix /= (all_max - all_min)
    text_list.append(word_vector_matrix)

batch = np.zeros((batch_size, max_sentence_size * max_word_size * 200))
batch = np.array(batch, dtype=np.float32)
output = np.zeros((batch_size, length_part_chapter_dict_wakati))
output = np.array(output, dtype=np.int32)

for b in range(batch_size):
    batch_before = np.zeros((max_word_size, 200, max_sentence_size))
    for i in range(len(text_list)):
        word_vector_matrix = text_list[i]
        dum_matrix = np.zeros((max_word_size, 200))
        dum_matrix[:word_vector_matrix.shape[0], :] = word_vector_matrix
        batch_before[:, :, i] = dum_matrix
    batch[b, :] = np.reshape(batch_before, (max_sentence_size * max_word_size * 200))
    
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
dum_zeros = tf.placeholder("float", shape=[None, max_word_size * 200])

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
            
# 畳み込み層1
W_conv1 = weight_variable([3, 3, 1, 64])
b_conv1 = bias_variable([64])

W_conv2 = weight_variable([3, 3, 64, 64])
b_conv2 = bias_variable([64])

W_conv3 = weight_variable([4, 4, 64, 128])
b_conv3 = bias_variable([128])

W_conv4 = weight_variable([4, 4, 128, 128])
b_conv4 = bias_variable([128])

W_fc1 = weight_variable([9 * 10 * 128, 128])
b_fc1 = bias_variable([128])

x_image = tf.reshape(x, [-1, max_word_size, 200, max_sentence_size])
z_image = tf.reshape(dum_zeros, [-1, max_word_size, 200, 1])

x_image0 = tf.reshape(x_image[:, :, :, 0], [-1, max_word_size, 200, 1])
h_conv1_0 = max_pool_2_2(tf.nn.relu(conv2d(max_pool_2_2(tf.nn.relu(conv2d(max_pool_2_2(tf.nn.relu(conv2d(max_pool_2_2(tf.nn.relu(conv2d(x_image0, W_conv1) + b_conv1)), W_conv2) + b_conv2)), W_conv3) + b_conv3)), W_conv4) + b_conv4))
h_flat1_0 = tf.reshape(h_conv1_0, [-1, 9 * 10 * 128])
h_fc1_0 = tf.nn.relu(tf.matmul(h_flat1_0, W_fc1) + b_fc1)

x_image1 = tf.reshape(x_image[:, :, :, 1], [-1, max_word_size, 200, 1])
h_conv1_1 = max_pool_2_2(tf.nn.relu(conv2d(max_pool_2_2(tf.nn.relu(conv2d(max_pool_2_2(tf.nn.relu(conv2d(max_pool_2_2(tf.nn.relu(conv2d(x_image1, W_conv1) + b_conv1)), W_conv2) + b_conv2)), W_conv3) + b_conv3)), W_conv4) + b_conv4))
h_flat1_1 = tf.reshape(h_conv1_1, [-1, 9 * 10 * 128])
h_fc1_1 = tf.nn.relu(tf.matmul(h_flat1_1, W_fc1) + b_fc1)

x_image2 = tf.reshape(x_image[:, :, :, 2], [-1, max_word_size, 200, 1])
h_conv1_2 = max_pool_2_2(tf.nn.relu(conv2d(max_pool_2_2(tf.nn.relu(conv2d(max_pool_2_2(tf.nn.relu(conv2d(max_pool_2_2(tf.nn.relu(conv2d(x_image2, W_conv1) + b_conv1)), W_conv2) + b_conv2)), W_conv3) + b_conv3)), W_conv4) + b_conv4))
h_flat1_2 = tf.reshape(h_conv1_2, [-1, 9 * 10 * 128])
h_fc1_2 = tf.nn.relu(tf.matmul(h_flat1_2, W_fc1) + b_fc1)

x_image3 = tf.reshape(x_image[:, :, :, 3], [-1, max_word_size, 200, 1])
h_conv1_3 = max_pool_2_2(tf.nn.relu(conv2d(max_pool_2_2(tf.nn.relu(conv2d(max_pool_2_2(tf.nn.relu(conv2d(max_pool_2_2(tf.nn.relu(conv2d(x_image3, W_conv1) + b_conv1)), W_conv2) + b_conv2)), W_conv3) + b_conv3)), W_conv4) + b_conv4))
h_flat1_3 = tf.reshape(h_conv1_3, [-1, 9 * 10 * 128])
h_fc1_3 = tf.nn.relu(tf.matmul(h_flat1_3, W_fc1) + b_fc1)

x_image4 = tf.reshape(x_image[:, :, :, 4], [-1, max_word_size, 200, 1])
h_conv1_4 = max_pool_2_2(tf.nn.relu(conv2d(max_pool_2_2(tf.nn.relu(conv2d(max_pool_2_2(tf.nn.relu(conv2d(max_pool_2_2(tf.nn.relu(conv2d(x_image4, W_conv1) + b_conv1)), W_conv2) + b_conv2)), W_conv3) + b_conv3)), W_conv4) + b_conv4))
h_flat1_4 = tf.reshape(h_conv1_4, [-1, 9 * 10 * 128])
h_fc1_4 = tf.nn.relu(tf.matmul(h_flat1_4, W_fc1) + b_fc1)

x_image5 = tf.reshape(x_image[:, :, :, 5], [-1, max_word_size, 200, 1])
h_conv1_5 = max_pool_2_2(tf.nn.relu(conv2d(max_pool_2_2(tf.nn.relu(conv2d(max_pool_2_2(tf.nn.relu(conv2d(max_pool_2_2(tf.nn.relu(conv2d(x_image5, W_conv1) + b_conv1)), W_conv2) + b_conv2)), W_conv3) + b_conv3)), W_conv4) + b_conv4))
h_flat1_5 = tf.reshape(h_conv1_5, [-1, 9 * 10 * 128])
h_fc1_5 = tf.nn.relu(tf.matmul(h_flat1_5, W_fc1) + b_fc1)

x_image6 = tf.reshape(x_image[:, :, :, 6], [-1, max_word_size, 200, 1])
h_conv1_6 = max_pool_2_2(tf.nn.relu(conv2d(max_pool_2_2(tf.nn.relu(conv2d(max_pool_2_2(tf.nn.relu(conv2d(max_pool_2_2(tf.nn.relu(conv2d(x_image6, W_conv1) + b_conv1)), W_conv2) + b_conv2)), W_conv3) + b_conv3)), W_conv4) + b_conv4))
h_flat1_6 = tf.reshape(h_conv1_6, [-1, 9 * 10 * 128])
h_fc1_6 = tf.nn.relu(tf.matmul(h_flat1_6, W_fc1) + b_fc1)

x_image7 = tf.reshape(x_image[:, :, :, 7], [-1, max_word_size, 200, 1])
h_conv1_7 = max_pool_2_2(tf.nn.relu(conv2d(max_pool_2_2(tf.nn.relu(conv2d(max_pool_2_2(tf.nn.relu(conv2d(max_pool_2_2(tf.nn.relu(conv2d(x_image7, W_conv1) + b_conv1)), W_conv2) + b_conv2)), W_conv3) + b_conv3)), W_conv4) + b_conv4))
h_flat1_7 = tf.reshape(h_conv1_7, [-1, 9 * 10 * 128])
h_fc1_7 = tf.nn.relu(tf.matmul(h_flat1_7, W_fc1) + b_fc1)

x_image8 = tf.reshape(x_image[:, :, :, 8], [-1, max_word_size, 200, 1])
h_conv1_8 = max_pool_2_2(tf.nn.relu(conv2d(max_pool_2_2(tf.nn.relu(conv2d(max_pool_2_2(tf.nn.relu(conv2d(max_pool_2_2(tf.nn.relu(conv2d(x_image8, W_conv1) + b_conv1)), W_conv2) + b_conv2)), W_conv3) + b_conv3)), W_conv4) + b_conv4))
h_flat1_8 = tf.reshape(h_conv1_8, [-1, 9 * 10 * 128])
h_fc1_8 = tf.nn.relu(tf.matmul(h_flat1_8, W_fc1) + b_fc1)

x_image9 = tf.reshape(x_image[:, :, :, 9], [-1, max_word_size, 200, 1])
h_conv1_9 = max_pool_2_2(tf.nn.relu(conv2d(max_pool_2_2(tf.nn.relu(conv2d(max_pool_2_2(tf.nn.relu(conv2d(max_pool_2_2(tf.nn.relu(conv2d(x_image9, W_conv1) + b_conv1)), W_conv2) + b_conv2)), W_conv3) + b_conv3)), W_conv4) + b_conv4))
h_flat1_9 = tf.reshape(h_conv1_9, [-1, 9 * 10 * 128])
h_fc1_9 = tf.nn.relu(tf.matmul(h_flat1_9, W_fc1) + b_fc1)

x_image10 = tf.reshape(x_image[:, :, :, 10], [-1, max_word_size, 200, 1])
h_conv1_10 = max_pool_2_2(tf.nn.relu(conv2d(max_pool_2_2(tf.nn.relu(conv2d(max_pool_2_2(tf.nn.relu(conv2d(max_pool_2_2(tf.nn.relu(conv2d(x_image10, W_conv1) + b_conv1)), W_conv2) + b_conv2)), W_conv3) + b_conv3)), W_conv4) + b_conv4))
h_flat1_10 = tf.reshape(h_conv1_10, [-1, 9 * 10 * 128])
h_fc1_10 = tf.nn.relu(tf.matmul(h_flat1_10, W_fc1) + b_fc1)

x_image11 = tf.reshape(x_image[:, :, :, 11], [-1, max_word_size, 200, 1])
h_conv1_11 = max_pool_2_2(tf.nn.relu(conv2d(max_pool_2_2(tf.nn.relu(conv2d(max_pool_2_2(tf.nn.relu(conv2d(max_pool_2_2(tf.nn.relu(conv2d(x_image11, W_conv1) + b_conv1)), W_conv2) + b_conv2)), W_conv3) + b_conv3)), W_conv4) + b_conv4))
h_flat1_11 = tf.reshape(h_conv1_11, [-1, 9 * 10 * 128])
h_fc1_11 = tf.nn.relu(tf.matmul(h_flat1_11, W_fc1) + b_fc1)

x_image12 = tf.reshape(x_image[:, :, :, 12], [-1, max_word_size, 200, 1])
h_conv1_12 = max_pool_2_2(tf.nn.relu(conv2d(max_pool_2_2(tf.nn.relu(conv2d(max_pool_2_2(tf.nn.relu(conv2d(max_pool_2_2(tf.nn.relu(conv2d(x_image12, W_conv1) + b_conv1)), W_conv2) + b_conv2)), W_conv3) + b_conv3)), W_conv4) + b_conv4))
h_flat1_12 = tf.reshape(h_conv1_12, [-1, 9 * 10 * 128])
h_fc1_12 = tf.nn.relu(tf.matmul(h_flat1_12, W_fc1) + b_fc1)

x_image13 = tf.reshape(x_image[:, :, :, 13], [-1, max_word_size, 200, 1])
h_conv1_13 = max_pool_2_2(tf.nn.relu(conv2d(max_pool_2_2(tf.nn.relu(conv2d(max_pool_2_2(tf.nn.relu(conv2d(max_pool_2_2(tf.nn.relu(conv2d(x_image13, W_conv1) + b_conv1)), W_conv2) + b_conv2)), W_conv3) + b_conv3)), W_conv4) + b_conv4))
h_flat1_13 = tf.reshape(h_conv1_13, [-1, 9 * 10 * 128])
h_fc1_13 = tf.nn.relu(tf.matmul(h_flat1_13, W_fc1) + b_fc1)

x_image14 = tf.reshape(x_image[:, :, :, 14], [-1, max_word_size, 200, 1])
h_conv1_14 = max_pool_2_2(tf.nn.relu(conv2d(max_pool_2_2(tf.nn.relu(conv2d(max_pool_2_2(tf.nn.relu(conv2d(max_pool_2_2(tf.nn.relu(conv2d(x_image14, W_conv1) + b_conv1)), W_conv2) + b_conv2)), W_conv3) + b_conv3)), W_conv4) + b_conv4))
h_flat1_14 = tf.reshape(h_conv1_14, [-1, 9 * 10 * 128])
h_fc1_14 = tf.nn.relu(tf.matmul(h_flat1_14, W_fc1) + b_fc1)

x_image15 = tf.reshape(x_image[:, :, :, 15], [-1, max_word_size, 200, 1])
h_conv1_15 = max_pool_2_2(tf.nn.relu(conv2d(max_pool_2_2(tf.nn.relu(conv2d(max_pool_2_2(tf.nn.relu(conv2d(max_pool_2_2(tf.nn.relu(conv2d(x_image15, W_conv1) + b_conv1)), W_conv2) + b_conv2)), W_conv3) + b_conv3)), W_conv4) + b_conv4))
h_flat1_15 = tf.reshape(h_conv1_15, [-1, 9 * 10 * 128])
h_fc1_15 = tf.nn.relu(tf.matmul(h_flat1_15, W_fc1) + b_fc1)

x_image16 = tf.reshape(x_image[:, :, :, 16], [-1, max_word_size, 200, 1])
h_conv1_16 = max_pool_2_2(tf.nn.relu(conv2d(max_pool_2_2(tf.nn.relu(conv2d(max_pool_2_2(tf.nn.relu(conv2d(max_pool_2_2(tf.nn.relu(conv2d(x_image16, W_conv1) + b_conv1)), W_conv2) + b_conv2)), W_conv3) + b_conv3)), W_conv4) + b_conv4))
h_flat1_16 = tf.reshape(h_conv1_16, [-1, 9 * 10 * 128])
h_fc1_16 = tf.nn.relu(tf.matmul(h_flat1_16, W_fc1) + b_fc1)

h_conv1_z = max_pool_2_2(tf.nn.relu(conv2d(max_pool_2_2(tf.nn.relu(conv2d(max_pool_2_2(tf.nn.relu(conv2d(max_pool_2_2(tf.nn.relu(conv2d(z_image, W_conv1) + b_conv1)), W_conv2) + b_conv2)), W_conv3) + b_conv3)), W_conv4) + b_conv4))
h_flat1_z = tf.reshape(h_conv1_z, [-1, 9 * 10 * 128])
h_fc1_z = tf.nn.relu(tf.matmul(h_flat1_z, W_fc1) + b_fc1)

r0 = tf.reshape(h_fc1_0, [-1, 1, 128])
r1 = tf.reshape(h_fc1_1, [-1, 1, 128])
r2 = tf.reshape(h_fc1_2, [-1, 1, 128])
r3 = tf.reshape(h_fc1_3, [-1, 1, 128])
r4 = tf.reshape(h_fc1_4, [-1, 1, 128])
r5 = tf.reshape(h_fc1_5, [-1, 1, 128])
r6 = tf.reshape(h_fc1_6, [-1, 1, 128])
r7 = tf.reshape(h_fc1_7, [-1, 1, 128])
r8 = tf.reshape(h_fc1_8, [-1, 1, 128])
r9 = tf.reshape(h_fc1_9, [-1, 1, 128])
r10 = tf.reshape(h_fc1_10, [-1, 1, 128])
r11 = tf.reshape(h_fc1_11, [-1, 1, 128])
r12 = tf.reshape(h_fc1_12, [-1, 1, 128])
r13 = tf.reshape(h_fc1_13, [-1, 1, 128])
r14 = tf.reshape(h_fc1_14, [-1, 1, 128])
r15 = tf.reshape(h_fc1_15, [-1, 1, 128])
r16 = tf.reshape(h_fc1_16, [-1, 1, 128])
rz = tf.reshape(h_fc1_z, [-1, 1, 128])

doc_matrix = tf.concat([r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15, r16], 1)
doc_matrix = tf.reshape(doc_matrix, [-1, max_sentence_size, 128, 1])

# 畳み込み層2
W_conv5 = weight_variable([3, 3, 1, 64])
b_conv5 = bias_variable([64])

W_conv6 = weight_variable([3, 3, 64, 64])
b_conv6 = bias_variable([64])

W_conv7 = weight_variable([3, 3, 64, 128])
b_conv7 = bias_variable([128])

h_conv2 = max_pool_2_2(tf.nn.relu(conv2d(max_pool_2_2(tf.nn.relu(conv2d(max_pool_2_2(tf.nn.relu(conv2d(doc_matrix, W_conv5) + b_conv5)), W_conv6) + b_conv6)), W_conv7) + b_conv7))
h_flat2 = tf.reshape(h_conv2, [-1, 1 * 15 * 128])

# 全結合層1
W_fc2 = weight_variable([128 * 15, 128])
b_fc2 = bias_variable([128])
h_fc2 = tf.nn.relu(tf.matmul(h_flat2, W_fc2) + b_fc2)

# 全結合層2
W_fc3 = weight_variable([128, 14])
b_fc3 = bias_variable([14])
y_out = tf.nn.softmax(tf.matmul(h_fc2, W_fc3) + b_fc3)

cross_entropy = -tf.reduce_sum(y_ * tf.log(tf.clip_by_value(y_out,1e-10,1.0)))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_out, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

doc_matrix0 = tf.concat([rz, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15, r16], 1)
doc_matrix0 = tf.reshape(doc_matrix0, [-1, max_sentence_size, 128, 1])
doc_matrix1 = tf.concat([r0, rz, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15, r16], 1)
doc_matrix1 = tf.reshape(doc_matrix1, [-1, max_sentence_size, 128, 1])
doc_matrix2 = tf.concat([r0, r1, rz, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15, r16], 1)
doc_matrix2 = tf.reshape(doc_matrix2, [-1, max_sentence_size, 128, 1])
doc_matrix3 = tf.concat([r0, r1, r2, rz, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15, r16], 1)
doc_matrix3 = tf.reshape(doc_matrix3, [-1, max_sentence_size, 128, 1])
doc_matrix4 = tf.concat([r0, r1, r2, r3, rz, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15, r16], 1)
doc_matrix4 = tf.reshape(doc_matrix4, [-1, max_sentence_size, 128, 1])
doc_matrix5 = tf.concat([r0, r1, r2, r3, r4, rz, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15, r16], 1)
doc_matrix5 = tf.reshape(doc_matrix5, [-1, max_sentence_size, 128, 1])
doc_matrix6 = tf.concat([r0, r1, r2, r3, r4, r5, rz, r7, r8, r9, r10, r11, r12, r13, r14, r15, r16], 1)
doc_matrix6 = tf.reshape(doc_matrix6, [-1, max_sentence_size, 128, 1])
doc_matrix7 = tf.concat([r0, r1, r2, r3, r4, r5, r6, rz, r8, r9, r10, r11, r12, r13, r14, r15, r16], 1)
doc_matrix7 = tf.reshape(doc_matrix7, [-1, max_sentence_size, 128, 1])
doc_matrix8 = tf.concat([r0, r1, r2, r3, r4, r5, r6, r7, rz, r9, r10, r11, r12, r13, r14, r15, r16], 1)
doc_matrix8 = tf.reshape(doc_matrix8, [-1, max_sentence_size, 128, 1])
doc_matrix9 = tf.concat([r0, r1, r2, r3, r4, r5, r6, r7, r8, rz, r10, r11, r12, r13, r14, r15, r16], 1)
doc_matrix9 = tf.reshape(doc_matrix9, [-1, max_sentence_size, 128, 1])
doc_matrix10 = tf.concat([r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, rz, r11, r12, r13, r14, r15, r16], 1)
doc_matrix10 = tf.reshape(doc_matrix10, [-1, max_sentence_size, 128, 1])
doc_matrix11 = tf.concat([r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, rz, r12, r13, r14, r15, r16], 1)
doc_matrix11 = tf.reshape(doc_matrix11, [-1, max_sentence_size, 128, 1])
doc_matrix12 = tf.concat([r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, rz, r13, r14, r15, r16], 1)
doc_matrix12 = tf.reshape(doc_matrix12, [-1, max_sentence_size, 128, 1])
doc_matrix13 = tf.concat([r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, rz, r14, r15, r16], 1)
doc_matrix13 = tf.reshape(doc_matrix13, [-1, max_sentence_size, 128, 1])
doc_matrix14 = tf.concat([r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, rz, r15, r16], 1)
doc_matrix14 = tf.reshape(doc_matrix14, [-1, max_sentence_size, 128, 1])
doc_matrix15 = tf.concat([r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, rz, r16], 1)
doc_matrix15 = tf.reshape(doc_matrix15, [-1, max_sentence_size, 128, 1])
doc_matrix16 = tf.concat([r0, r1, r2, r3, r4, r5, r6, r7, r8, r9, r10, r11, r12, r13, r14, r15, rz], 1)
doc_matrix16 = tf.reshape(doc_matrix16, [-1, max_sentence_size, 128, 1])

h_conv2_0 = max_pool_2_2(tf.nn.relu(conv2d(max_pool_2_2(tf.nn.relu(conv2d(max_pool_2_2(tf.nn.relu(conv2d(doc_matrix0, W_conv5) + b_conv5)), W_conv6) + b_conv6)), W_conv7) + b_conv7))
h_flat2_0 = tf.reshape(h_conv2_0, [-1, 1 * 15 * 128])
h_fc2_0 = tf.nn.relu(tf.matmul(h_flat2_0, W_fc2) + b_fc2)
y_out_0 = tf.nn.softmax(tf.matmul(h_fc2_0, W_fc3) + b_fc3)
loss0 = -tf.reduce_sum(y_ * tf.log(tf.clip_by_value(y_out_0,1e-10,1.0)))

h_conv2_1 = max_pool_2_2(tf.nn.relu(conv2d(max_pool_2_2(tf.nn.relu(conv2d(max_pool_2_2(tf.nn.relu(conv2d(doc_matrix1, W_conv5) + b_conv5)), W_conv6) + b_conv6)), W_conv7) + b_conv7))
h_flat2_1 = tf.reshape(h_conv2_1, [-1, 1 * 15 * 128])
h_fc2_1 = tf.nn.relu(tf.matmul(h_flat2_1, W_fc2) + b_fc2)
y_out_1 = tf.nn.softmax(tf.matmul(h_fc2_1, W_fc3) + b_fc3)
loss1 = -tf.reduce_sum(y_ * tf.log(tf.clip_by_value(y_out_1,1e-10,1.0)))

h_conv2_2 = max_pool_2_2(tf.nn.relu(conv2d(max_pool_2_2(tf.nn.relu(conv2d(max_pool_2_2(tf.nn.relu(conv2d(doc_matrix2, W_conv5) + b_conv5)), W_conv6) + b_conv6)), W_conv7) + b_conv7))
h_flat2_2 = tf.reshape(h_conv2_2, [-1, 1 * 15 * 128])
h_fc2_2 = tf.nn.relu(tf.matmul(h_flat2_2, W_fc2) + b_fc2)
y_out_2 = tf.nn.softmax(tf.matmul(h_fc2_2, W_fc3) + b_fc3)
loss2 = -tf.reduce_sum(y_ * tf.log(tf.clip_by_value(y_out_2,1e-10,1.0)))

h_conv2_3 = max_pool_2_2(tf.nn.relu(conv2d(max_pool_2_2(tf.nn.relu(conv2d(max_pool_2_2(tf.nn.relu(conv2d(doc_matrix3, W_conv5) + b_conv5)), W_conv6) + b_conv6)), W_conv7) + b_conv7))
h_flat2_3 = tf.reshape(h_conv2_3, [-1, 1 * 15 * 128])
h_fc2_3 = tf.nn.relu(tf.matmul(h_flat2_3, W_fc2) + b_fc2)
y_out_3 = tf.nn.softmax(tf.matmul(h_fc2_3, W_fc3) + b_fc3)
loss3 = -tf.reduce_sum(y_ * tf.log(tf.clip_by_value(y_out_3,1e-10,1.0)))

h_conv2_4 = max_pool_2_2(tf.nn.relu(conv2d(max_pool_2_2(tf.nn.relu(conv2d(max_pool_2_2(tf.nn.relu(conv2d(doc_matrix4, W_conv5) + b_conv5)), W_conv6) + b_conv6)), W_conv7) + b_conv7))
h_flat2_4 = tf.reshape(h_conv2_4, [-1, 1 * 15 * 128])
h_fc2_4 = tf.nn.relu(tf.matmul(h_flat2_4, W_fc2) + b_fc2)
y_out_4 = tf.nn.softmax(tf.matmul(h_fc2_4, W_fc3) + b_fc3)
loss4 = -tf.reduce_sum(y_ * tf.log(tf.clip_by_value(y_out_4,1e-10,1.0)))

h_conv2_5 = max_pool_2_2(tf.nn.relu(conv2d(max_pool_2_2(tf.nn.relu(conv2d(max_pool_2_2(tf.nn.relu(conv2d(doc_matrix5, W_conv5) + b_conv5)), W_conv6) + b_conv6)), W_conv7) + b_conv7))
h_flat2_5 = tf.reshape(h_conv2_5, [-1, 1 * 15 * 128])
h_fc2_5 = tf.nn.relu(tf.matmul(h_flat2_5, W_fc2) + b_fc2)
y_out_5 = tf.nn.softmax(tf.matmul(h_fc2_5, W_fc3) + b_fc3)
loss5 = -tf.reduce_sum(y_ * tf.log(tf.clip_by_value(y_out_5,1e-10,1.0)))

h_conv2_6 = max_pool_2_2(tf.nn.relu(conv2d(max_pool_2_2(tf.nn.relu(conv2d(max_pool_2_2(tf.nn.relu(conv2d(doc_matrix6, W_conv5) + b_conv5)), W_conv6) + b_conv6)), W_conv7) + b_conv7))
h_flat2_6 = tf.reshape(h_conv2_6, [-1, 1 * 15 * 128])
h_fc2_6 = tf.nn.relu(tf.matmul(h_flat2_6, W_fc2) + b_fc2)
y_out_6 = tf.nn.softmax(tf.matmul(h_fc2_6, W_fc3) + b_fc3)
loss6 = -tf.reduce_sum(y_ * tf.log(tf.clip_by_value(y_out_6,1e-10,1.0)))

h_conv2_7 = max_pool_2_2(tf.nn.relu(conv2d(max_pool_2_2(tf.nn.relu(conv2d(max_pool_2_2(tf.nn.relu(conv2d(doc_matrix7, W_conv5) + b_conv5)), W_conv6) + b_conv6)), W_conv7) + b_conv7))
h_flat2_7 = tf.reshape(h_conv2_7, [-1, 1 * 15 * 128])
h_fc2_7 = tf.nn.relu(tf.matmul(h_flat2_7, W_fc2) + b_fc2)
y_out_7 = tf.nn.softmax(tf.matmul(h_fc2_7, W_fc3) + b_fc3)
loss7 = -tf.reduce_sum(y_ * tf.log(tf.clip_by_value(y_out_7,1e-10,1.0)))

h_conv2_8 = max_pool_2_2(tf.nn.relu(conv2d(max_pool_2_2(tf.nn.relu(conv2d(max_pool_2_2(tf.nn.relu(conv2d(doc_matrix8, W_conv5) + b_conv5)), W_conv6) + b_conv6)), W_conv7) + b_conv7))
h_flat2_8 = tf.reshape(h_conv2_8, [-1, 1 * 15 * 128])
h_fc2_8 = tf.nn.relu(tf.matmul(h_flat2_8, W_fc2) + b_fc2)
y_out_8 = tf.nn.softmax(tf.matmul(h_fc2_8, W_fc3) + b_fc3)
loss8 = -tf.reduce_sum(y_ * tf.log(tf.clip_by_value(y_out_8,1e-10,1.0)))

h_conv2_9 = max_pool_2_2(tf.nn.relu(conv2d(max_pool_2_2(tf.nn.relu(conv2d(max_pool_2_2(tf.nn.relu(conv2d(doc_matrix9, W_conv5) + b_conv5)), W_conv6) + b_conv6)), W_conv7) + b_conv7))
h_flat2_9 = tf.reshape(h_conv2_9, [-1, 1 * 15 * 128])
h_fc2_9 = tf.nn.relu(tf.matmul(h_flat2_9, W_fc2) + b_fc2)
y_out_9 = tf.nn.softmax(tf.matmul(h_fc2_9, W_fc3) + b_fc3)
loss9 = -tf.reduce_sum(y_ * tf.log(tf.clip_by_value(y_out_9,1e-10,1.0)))

h_conv2_10 = max_pool_2_2(tf.nn.relu(conv2d(max_pool_2_2(tf.nn.relu(conv2d(max_pool_2_2(tf.nn.relu(conv2d(doc_matrix10, W_conv5) + b_conv5)), W_conv6) + b_conv6)), W_conv7) + b_conv7))
h_flat2_10 = tf.reshape(h_conv2_10, [-1, 1 * 15 * 128])
h_fc2_10 = tf.nn.relu(tf.matmul(h_flat2_10, W_fc2) + b_fc2)
y_out_10 = tf.nn.softmax(tf.matmul(h_fc2_10, W_fc3) + b_fc3)
loss10 = -tf.reduce_sum(y_ * tf.log(tf.clip_by_value(y_out_10,1e-10,1.0)))

h_conv2_11 = max_pool_2_2(tf.nn.relu(conv2d(max_pool_2_2(tf.nn.relu(conv2d(max_pool_2_2(tf.nn.relu(conv2d(doc_matrix11, W_conv5) + b_conv5)), W_conv6) + b_conv6)), W_conv7) + b_conv7))
h_flat2_11 = tf.reshape(h_conv2_11, [-1, 1 * 15 * 128])
h_fc2_11 = tf.nn.relu(tf.matmul(h_flat2_11, W_fc2) + b_fc2)
y_out_11 = tf.nn.softmax(tf.matmul(h_fc2_11, W_fc3) + b_fc3)
loss11 = -tf.reduce_sum(y_ * tf.log(tf.clip_by_value(y_out_11,1e-10,1.0)))

h_conv2_12 = max_pool_2_2(tf.nn.relu(conv2d(max_pool_2_2(tf.nn.relu(conv2d(max_pool_2_2(tf.nn.relu(conv2d(doc_matrix12, W_conv5) + b_conv5)), W_conv6) + b_conv6)), W_conv7) + b_conv7))
h_flat2_12 = tf.reshape(h_conv2_12, [-1, 1 * 15 * 128])
h_fc2_12 = tf.nn.relu(tf.matmul(h_flat2_12, W_fc2) + b_fc2)
y_out_12 = tf.nn.softmax(tf.matmul(h_fc2_12, W_fc3) + b_fc3)
loss12 = -tf.reduce_sum(y_ * tf.log(tf.clip_by_value(y_out_12,1e-10,1.0)))

h_conv2_13 = max_pool_2_2(tf.nn.relu(conv2d(max_pool_2_2(tf.nn.relu(conv2d(max_pool_2_2(tf.nn.relu(conv2d(doc_matrix13, W_conv5) + b_conv5)), W_conv6) + b_conv6)), W_conv7) + b_conv7))
h_flat2_13 = tf.reshape(h_conv2_13, [-1, 1 * 15 * 128])
h_fc2_13 = tf.nn.relu(tf.matmul(h_flat2_13, W_fc2) + b_fc2)
y_out_13 = tf.nn.softmax(tf.matmul(h_fc2_13, W_fc3) + b_fc3)
loss13 = -tf.reduce_sum(y_ * tf.log(tf.clip_by_value(y_out_13,1e-10,1.0)))

h_conv2_14 = max_pool_2_2(tf.nn.relu(conv2d(max_pool_2_2(tf.nn.relu(conv2d(max_pool_2_2(tf.nn.relu(conv2d(doc_matrix14, W_conv5) + b_conv5)), W_conv6) + b_conv6)), W_conv7) + b_conv7))
h_flat2_14 = tf.reshape(h_conv2_14, [-1, 1 * 15 * 128])
h_fc2_14 = tf.nn.relu(tf.matmul(h_flat2_14, W_fc2) + b_fc2)
y_out_14 = tf.nn.softmax(tf.matmul(h_fc2_14, W_fc3) + b_fc3)
loss14 = -tf.reduce_sum(y_ * tf.log(tf.clip_by_value(y_out_14,1e-10,1.0)))

h_conv2_15 = max_pool_2_2(tf.nn.relu(conv2d(max_pool_2_2(tf.nn.relu(conv2d(max_pool_2_2(tf.nn.relu(conv2d(doc_matrix15, W_conv5) + b_conv5)), W_conv6) + b_conv6)), W_conv7) + b_conv7))
h_flat2_15 = tf.reshape(h_conv2_15, [-1, 1 * 15 * 128])
h_fc2_15 = tf.nn.relu(tf.matmul(h_flat2_15, W_fc2) + b_fc2)
y_out_15 = tf.nn.softmax(tf.matmul(h_fc2_15, W_fc3) + b_fc3)
loss15 = -tf.reduce_sum(y_ * tf.log(tf.clip_by_value(y_out_15,1e-10,1.0)))

h_conv2_16 = max_pool_2_2(tf.nn.relu(conv2d(max_pool_2_2(tf.nn.relu(conv2d(max_pool_2_2(tf.nn.relu(conv2d(doc_matrix16, W_conv5) + b_conv5)), W_conv6) + b_conv6)), W_conv7) + b_conv7))
h_flat2_16 = tf.reshape(h_conv2_16, [-1, 1 * 15 * 128])
h_fc2_16 = tf.nn.relu(tf.matmul(h_flat2_16, W_fc2) + b_fc2)
y_out_16 = tf.nn.softmax(tf.matmul(h_fc2_16, W_fc3) + b_fc3)
loss16 = -tf.reduce_sum(y_ * tf.log(tf.clip_by_value(y_out_16,1e-10,1.0)))

sess = tf.Session()
saver = tf.train.Saver()
saver.restore(sess, "./20190114word_sentence_document2")

batch_zeros = np.zeros((batch_size, max_word_size * 200))
batch_zeros = np.array(batch_zeros, dtype=np.float32)

out = y_out.eval(session=sess, feed_dict={x: batch, y_: output, dum_zeros: batch_zeros})
output = out

loss = cross_entropy.eval(session=sess, feed_dict={x: batch, y_: output, dum_zeros: batch_zeros})

loss_list = []
loss_0 = loss0.eval(session=sess, feed_dict={x: batch, y_: output, dum_zeros: batch_zeros})
loss_list.append(loss_0)
loss_1 = loss1.eval(session=sess, feed_dict={x: batch, y_: output, dum_zeros: batch_zeros})
loss_list.append(loss_1)
loss_2 = loss2.eval(session=sess, feed_dict={x: batch, y_: output, dum_zeros: batch_zeros})
loss_list.append(loss_2)
loss_3 = loss3.eval(session=sess, feed_dict={x: batch, y_: output, dum_zeros: batch_zeros})
loss_list.append(loss_3)
loss_4 = loss4.eval(session=sess, feed_dict={x: batch, y_: output, dum_zeros: batch_zeros})
loss_list.append(loss_4)
loss_5 = loss5.eval(session=sess, feed_dict={x: batch, y_: output, dum_zeros: batch_zeros})
loss_list.append(loss_5)
loss_6 = loss6.eval(session=sess, feed_dict={x: batch, y_: output, dum_zeros: batch_zeros})
loss_list.append(loss_6)
loss_7 = loss7.eval(session=sess, feed_dict={x: batch, y_: output, dum_zeros: batch_zeros})
loss_list.append(loss_7)
loss_8 = loss8.eval(session=sess, feed_dict={x: batch, y_: output, dum_zeros: batch_zeros})
loss_list.append(loss_8)
loss_9 = loss9.eval(session=sess, feed_dict={x: batch, y_: output, dum_zeros: batch_zeros})
loss_list.append(loss_9)
loss_10 = loss10.eval(session=sess, feed_dict={x: batch, y_: output, dum_zeros: batch_zeros})
loss_list.append(loss_10)
loss_11 = loss11.eval(session=sess, feed_dict={x: batch, y_: output, dum_zeros: batch_zeros})
loss_list.append(loss_11)
loss_12 = loss12.eval(session=sess, feed_dict={x: batch, y_: output, dum_zeros: batch_zeros})
loss_list.append(loss_12)
loss_13 = loss13.eval(session=sess, feed_dict={x: batch, y_: output, dum_zeros: batch_zeros})
loss_list.append(loss_13)
loss_14 = loss14.eval(session=sess, feed_dict={x: batch, y_: output, dum_zeros: batch_zeros})
loss_list.append(loss_14)
loss_15 = loss15.eval(session=sess, feed_dict={x: batch, y_: output, dum_zeros: batch_zeros})
loss_list.append(loss_15)
loss_16 = loss16.eval(session=sess, feed_dict={x: batch, y_: output, dum_zeros: batch_zeros})
loss_list.append(loss_16)

sess.close()

def add_contribution_score(t_l, a_l):
    s_l = []
    for t in range(len(t_l)):
        s_l.append((loss_list[t] / max(loss_list)))
    a_l.append(s_l)
    return a_l

def add_filter_figure_title(t_l, a_l):
    s_l = []
    for t in range(len(t_l)):
        s_l.append(0)
    for t in range(len(t_l)):
        for w in figure_title_word_list:
            if w in t_l[t]:
                s_l[t] = s_l[t] + 1
    t_s_l = []
    for s in s_l:
        t_s_l.append((s / max(s_l)))
    a_l.append(t_s_l)
    return a_l

def add_filter_part_title(t_l, a_l):
    s_l = []
    for t in range(len(t_l)):
        s_l.append(0)
    for t in range(len(t_l)):
        for w in filter_part_title_list:
            if w in t_l[t]:
                s_l[t] = s_l[t] + 1
    t_s_l = []
    for s in s_l:
        t_s_l.append((s / max(s_l)))
    a_l.append(t_s_l)
    return a_l

def add_filter_frequent_word(t_l, a_l):
    s_l = []
    for t in range(len(t_l)):
        s_l.append(0)
    for t in range(len(t_l)):
        for k in frequent_dict:
            if k in t_l[t]:
                s_l[t] = s_l[t] + 1
    t_s_l = []
    for s in s_l:
        t_s_l.append((s / max(s_l)))
    a_l.append(t_s_l)
    return a_l

def add_filter_connection_word(t_l, a_l):
    s_l = []
    for t in range(len(t_l)):
        s_l.append(0)
    for t in range(len(t_l)):
        for w in connection_word_list:
            if w in t_l[t]:
                s_l[t] = s_l[t] + 1
    t_s_l = []
    for s in s_l:
        if max(s_l) == 0.0:
            t_s_l.append(s)
        else:
            t_s_l.append((s / max(s_l)))
    a_l.append(t_s_l)
    return a_l

def add_filter_position(t_l, a_l):
    s_l = []
    for t in range(len(t_l)):
        s_l.append(0)
    s_l[0] = 0.5
    s_l[len(t_l)-1] = 0.5
    a_l.append(s_l)
    return a_l

def calc_all_score(a_l, f_s):
    t_a_l = []
    for i in range(len(a_l[0])):
        score = 0.0
        for s_l in a_l:
            score += s_l[i] / f_s
        t_a_l.append(score)
    return t_a_l
        

text_list = []
for t in range(len(texts)):
    if loss_list[t] != 0.0:
        text_list.append(texts[t])

all_score_list = []
all_score_list = add_contribution_score(text_list, all_score_list)
filter_size = 1.0
if filter_figure_title:
    all_score_list = add_filter_figure_title(text_list, all_score_list)
    filter_size += 1.0
if filter_part_title:
    all_score_list = add_filter_part_title(text_list, all_score_list)
    filter_size += 1.0
if filter_frequent_word:
    all_score_list = add_filter_frequent_word(text_list, all_score_list)
    filter_size += 1.0
if filter_connection_word:
    all_score_list = add_filter_connection_word(text_list, all_score_list)
    filter_size += 1.0
if filter_position:
    all_score_list = add_filter_position(text_list, all_score_list)
    filter_size += 0.5

all_score_list = calc_all_score(all_score_list, filter_size)
all_score_list = np.array(all_score_list)
max_sort_indexes = np.argsort(all_score_list)[::-1]

for i in range(len(all_score_list)):
    index = max_sort_indexes[i]
    print(texts[index])
    print(all_score_list[index])