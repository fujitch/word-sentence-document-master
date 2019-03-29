# -*- coding: utf-8 -*-

import MeCab
from gensim.models import word2vec
from tensorflow.contrib.framework import sort
import pickle
import numpy as np
import tensorflow as tf

m_owakati = MeCab.Tagger(r'-Owakati -d C:\Users\hori\workspace\encoder-decoder-sentence-chainer-master\mecab-ipadic-neologd')
model = pickle.load(open('word2vec_neologd30.pickle', 'rb'))
part_chapter_dict = pickle.load(open("energy_paper_2018_dict.pickle", "rb"))

texts = "日本のガス事業は、1872年10月31日（旧暦9月29日）に、横浜の馬車道にガス灯が点灯したことから始まりました。神奈川県庁付近および大江橋から馬車道・本町通りまでの間にガス灯十数基が点灯され、日本で初めての近代的照明となったガス灯の点灯は、産業近代化を象徴するものでもありました。ガス灯が点灯された当日は、横浜市民だけではなく、東京方面からも多くの見物人が訪れ、祭りのような賑わいになりました。"
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
    text_list.append(word_vector_matrix)

part_chapter_dict_wakati = pickle.load(open("part_chapter_dict_wakati.pickle", "rb"))
length_part_chapter_dict_wakati = len(part_chapter_dict_wakati)
batch_size = len(part_chapter_dict_wakati)

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

tf.reset_default_graph()

x = tf.placeholder("float", shape=[None, max_sentence_size * max_word_size * 200])
y_ = tf.placeholder("float", shape=[None, len(part_chapter_dict_wakati)])
sentence_num = tf.placeholder("float", shape=[1])

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

# プーリング処理を定義
def max_pool_2_2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')
            
# 畳み込み層1
W_conv1 = weight_variable([3, 3, 1, 256])
b_conv1 = bias_variable([256])

W_conv2 = weight_variable([3, 3, 256, 256])
b_conv2 = bias_variable([256])

W_conv3 = weight_variable([4, 4, 256, 256])
b_conv3 = bias_variable([256])

W_conv3 = weight_variable([4, 4, 256, 256])
b_conv3 = bias_variable([256])

W_conv4 = weight_variable([4, 4, 256, 256])
b_conv4 = bias_variable([256])

W_fc1 = weight_variable([9 * 10 * 256, 128])
b_fc1 = bias_variable([128])

x_image = tf.reshape(x, [-1, max_word_size, 200, max_sentence_size])
z_image = tf.reshape(dum_zeros, [-1, max_word_size, 200, 1])

x_image0 = tf.reshape(x_image[:, :, :, 0], [-1, max_word_size, 200, 1])
h_conv1_0 = max_pool_2_2(tf.nn.relu(conv2d(max_pool_2_2(tf.nn.relu(conv2d(max_pool_2_2(tf.nn.relu(conv2d(max_pool_2_2(tf.nn.relu(conv2d(x_image0, W_conv1) + b_conv1)), W_conv2) + b_conv2)), W_conv3) + b_conv3)), W_conv4) + b_conv4))
h_flat1_0 = tf.reshape(h_conv1_0, [-1, 9 * 10 * 256])
h_fc1_0 = tf.nn.relu(tf.matmul(h_flat1_0, W_fc1) + b_fc1)

x_image1 = tf.reshape(x_image[:, :, :, 1], [-1, max_word_size, 200, 1])
h_conv1_1 = max_pool_2_2(tf.nn.relu(conv2d(max_pool_2_2(tf.nn.relu(conv2d(max_pool_2_2(tf.nn.relu(conv2d(max_pool_2_2(tf.nn.relu(conv2d(x_image1, W_conv1) + b_conv1)), W_conv2) + b_conv2)), W_conv3) + b_conv3)), W_conv4) + b_conv4))
h_flat1_1 = tf.reshape(h_conv1_1, [-1, 9 * 10 * 256])
h_fc1_1 = tf.nn.relu(tf.matmul(h_flat1_1, W_fc1) + b_fc1)

x_image2 = tf.reshape(x_image[:, :, :, 2], [-1, max_word_size, 200, 1])
h_conv1_2 = max_pool_2_2(tf.nn.relu(conv2d(max_pool_2_2(tf.nn.relu(conv2d(max_pool_2_2(tf.nn.relu(conv2d(max_pool_2_2(tf.nn.relu(conv2d(x_image2, W_conv1) + b_conv1)), W_conv2) + b_conv2)), W_conv3) + b_conv3)), W_conv4) + b_conv4))
h_flat1_2 = tf.reshape(h_conv1_2, [-1, 9 * 10 * 256])
h_fc1_2 = tf.nn.relu(tf.matmul(h_flat1_2, W_fc1) + b_fc1)

x_image3 = tf.reshape(x_image[:, :, :, 3], [-1, max_word_size, 200, 1])
h_conv1_3 = max_pool_2_2(tf.nn.relu(conv2d(max_pool_2_2(tf.nn.relu(conv2d(max_pool_2_2(tf.nn.relu(conv2d(max_pool_2_2(tf.nn.relu(conv2d(x_image3, W_conv1) + b_conv1)), W_conv2) + b_conv2)), W_conv3) + b_conv3)), W_conv4) + b_conv4))
h_flat1_3 = tf.reshape(h_conv1_3, [-1, 9 * 10 * 256])
h_fc1_3 = tf.nn.relu(tf.matmul(h_flat1_3, W_fc1) + b_fc1)

x_image4 = tf.reshape(x_image[:, :, :, 4], [-1, max_word_size, 200, 1])
h_conv1_4 = max_pool_2_2(tf.nn.relu(conv2d(max_pool_2_2(tf.nn.relu(conv2d(max_pool_2_2(tf.nn.relu(conv2d(max_pool_2_2(tf.nn.relu(conv2d(x_image4, W_conv1) + b_conv1)), W_conv2) + b_conv2)), W_conv3) + b_conv3)), W_conv4) + b_conv4))
h_flat1_4 = tf.reshape(h_conv1_4, [-1, 9 * 10 * 256])
h_fc1_4 = tf.nn.relu(tf.matmul(h_flat1_4, W_fc1) + b_fc1)

x_image5 = tf.reshape(x_image[:, :, :, 5], [-1, max_word_size, 200, 1])
h_conv1_5 = max_pool_2_2(tf.nn.relu(conv2d(max_pool_2_2(tf.nn.relu(conv2d(max_pool_2_2(tf.nn.relu(conv2d(max_pool_2_2(tf.nn.relu(conv2d(x_image5, W_conv1) + b_conv1)), W_conv2) + b_conv2)), W_conv3) + b_conv3)), W_conv4) + b_conv4))
h_flat1_5 = tf.reshape(h_conv1_5, [-1, 9 * 10 * 256])
h_fc1_5 = tf.nn.relu(tf.matmul(h_flat1_5, W_fc1) + b_fc1)

x_image6 = tf.reshape(x_image[:, :, :, 6], [-1, max_word_size, 200, 1])
h_conv1_6 = max_pool_2_2(tf.nn.relu(conv2d(max_pool_2_2(tf.nn.relu(conv2d(max_pool_2_2(tf.nn.relu(conv2d(max_pool_2_2(tf.nn.relu(conv2d(x_image6, W_conv1) + b_conv1)), W_conv2) + b_conv2)), W_conv3) + b_conv3)), W_conv4) + b_conv4))
h_flat1_6 = tf.reshape(h_conv1_6, [-1, 9 * 10 * 256])
h_fc1_6 = tf.nn.relu(tf.matmul(h_flat1_6, W_fc1) + b_fc1)

x_image7 = tf.reshape(x_image[:, :, :, 7], [-1, max_word_size, 200, 1])
h_conv1_7 = max_pool_2_2(tf.nn.relu(conv2d(max_pool_2_2(tf.nn.relu(conv2d(max_pool_2_2(tf.nn.relu(conv2d(max_pool_2_2(tf.nn.relu(conv2d(x_image7, W_conv1) + b_conv1)), W_conv2) + b_conv2)), W_conv3) + b_conv3)), W_conv4) + b_conv4))
h_flat1_7 = tf.reshape(h_conv1_7, [-1, 9 * 10 * 256])
h_fc1_7 = tf.nn.relu(tf.matmul(h_flat1_7, W_fc1) + b_fc1)

x_image8 = tf.reshape(x_image[:, :, :, 8], [-1, max_word_size, 200, 1])
h_conv1_8 = max_pool_2_2(tf.nn.relu(conv2d(max_pool_2_2(tf.nn.relu(conv2d(max_pool_2_2(tf.nn.relu(conv2d(max_pool_2_2(tf.nn.relu(conv2d(x_image8, W_conv1) + b_conv1)), W_conv2) + b_conv2)), W_conv3) + b_conv3)), W_conv4) + b_conv4))
h_flat1_8 = tf.reshape(h_conv1_8, [-1, 9 * 10 * 256])
h_fc1_8 = tf.nn.relu(tf.matmul(h_flat1_8, W_fc1) + b_fc1)

x_image9 = tf.reshape(x_image[:, :, :, 9], [-1, max_word_size, 200, 1])
h_conv1_9 = max_pool_2_2(tf.nn.relu(conv2d(max_pool_2_2(tf.nn.relu(conv2d(max_pool_2_2(tf.nn.relu(conv2d(max_pool_2_2(tf.nn.relu(conv2d(x_image9, W_conv1) + b_conv1)), W_conv2) + b_conv2)), W_conv3) + b_conv3)), W_conv4) + b_conv4))
h_flat1_9 = tf.reshape(h_conv1_9, [-1, 9 * 10 * 256])
h_fc1_9 = tf.nn.relu(tf.matmul(h_flat1_9, W_fc1) + b_fc1)

x_image10 = tf.reshape(x_image[:, :, :, 10], [-1, max_word_size, 200, 1])
h_conv1_10 = max_pool_2_2(tf.nn.relu(conv2d(max_pool_2_2(tf.nn.relu(conv2d(max_pool_2_2(tf.nn.relu(conv2d(max_pool_2_2(tf.nn.relu(conv2d(x_image10, W_conv1) + b_conv1)), W_conv2) + b_conv2)), W_conv3) + b_conv3)), W_conv4) + b_conv4))
h_flat1_10 = tf.reshape(h_conv1_10, [-1, 9 * 10 * 256])
h_fc1_10 = tf.nn.relu(tf.matmul(h_flat1_10, W_fc1) + b_fc1)

x_image11 = tf.reshape(x_image[:, :, :, 11], [-1, max_word_size, 200, 1])
h_conv1_11 = max_pool_2_2(tf.nn.relu(conv2d(max_pool_2_2(tf.nn.relu(conv2d(max_pool_2_2(tf.nn.relu(conv2d(max_pool_2_2(tf.nn.relu(conv2d(x_image11, W_conv1) + b_conv1)), W_conv2) + b_conv2)), W_conv3) + b_conv3)), W_conv4) + b_conv4))
h_flat1_11 = tf.reshape(h_conv1_11, [-1, 9 * 10 * 256])
h_fc1_11 = tf.nn.relu(tf.matmul(h_flat1_11, W_fc1) + b_fc1)

x_image12 = tf.reshape(x_image[:, :, :, 12], [-1, max_word_size, 200, 1])
h_conv1_12 = max_pool_2_2(tf.nn.relu(conv2d(max_pool_2_2(tf.nn.relu(conv2d(max_pool_2_2(tf.nn.relu(conv2d(max_pool_2_2(tf.nn.relu(conv2d(x_image12, W_conv1) + b_conv1)), W_conv2) + b_conv2)), W_conv3) + b_conv3)), W_conv4) + b_conv4))
h_flat1_12 = tf.reshape(h_conv1_12, [-1, 9 * 10 * 256])
h_fc1_12 = tf.nn.relu(tf.matmul(h_flat1_12, W_fc1) + b_fc1)

x_image13 = tf.reshape(x_image[:, :, :, 13], [-1, max_word_size, 200, 1])
h_conv1_13 = max_pool_2_2(tf.nn.relu(conv2d(max_pool_2_2(tf.nn.relu(conv2d(max_pool_2_2(tf.nn.relu(conv2d(max_pool_2_2(tf.nn.relu(conv2d(x_image13, W_conv1) + b_conv1)), W_conv2) + b_conv2)), W_conv3) + b_conv3)), W_conv4) + b_conv4))
h_flat1_13 = tf.reshape(h_conv1_13, [-1, 9 * 10 * 256])
h_fc1_13 = tf.nn.relu(tf.matmul(h_flat1_13, W_fc1) + b_fc1)

x_image14 = tf.reshape(x_image[:, :, :, 14], [-1, max_word_size, 200, 1])
h_conv1_14 = max_pool_2_2(tf.nn.relu(conv2d(max_pool_2_2(tf.nn.relu(conv2d(max_pool_2_2(tf.nn.relu(conv2d(max_pool_2_2(tf.nn.relu(conv2d(x_image14, W_conv1) + b_conv1)), W_conv2) + b_conv2)), W_conv3) + b_conv3)), W_conv4) + b_conv4))
h_flat1_14 = tf.reshape(h_conv1_14, [-1, 9 * 10 * 256])
h_fc1_14 = tf.nn.relu(tf.matmul(h_flat1_14, W_fc1) + b_fc1)

x_image15 = tf.reshape(x_image[:, :, :, 15], [-1, max_word_size, 200, 1])
h_conv1_15 = max_pool_2_2(tf.nn.relu(conv2d(max_pool_2_2(tf.nn.relu(conv2d(max_pool_2_2(tf.nn.relu(conv2d(max_pool_2_2(tf.nn.relu(conv2d(x_image15, W_conv1) + b_conv1)), W_conv2) + b_conv2)), W_conv3) + b_conv3)), W_conv4) + b_conv4))
h_flat1_15 = tf.reshape(h_conv1_15, [-1, 9 * 10 * 256])
h_fc1_15 = tf.nn.relu(tf.matmul(h_flat1_15, W_fc1) + b_fc1)

x_image16 = tf.reshape(x_image[:, :, :, 16], [-1, max_word_size, 200, 1])
h_conv1_16 = max_pool_2_2(tf.nn.relu(conv2d(max_pool_2_2(tf.nn.relu(conv2d(max_pool_2_2(tf.nn.relu(conv2d(max_pool_2_2(tf.nn.relu(conv2d(x_image16, W_conv1) + b_conv1)), W_conv2) + b_conv2)), W_conv3) + b_conv3)), W_conv4) + b_conv4))
h_flat1_16 = tf.reshape(h_conv1_16, [-1, 9 * 10 * 256])
h_fc1_16 = tf.nn.relu(tf.matmul(h_flat1_16, W_fc1) + b_fc1)

h_conv1_z = max_pool_2_2(tf.nn.relu(conv2d(max_pool_2_2(tf.nn.relu(conv2d(max_pool_2_2(tf.nn.relu(conv2d(max_pool_2_2(tf.nn.relu(conv2d(z_image, W_conv1) + b_conv1)), W_conv2) + b_conv2)), W_conv3) + b_conv3)), W_conv4) + b_conv4))
h_flat1_z = tf.reshape(h_conv1_z, [-1, 9 * 10 * 256])
h_fc1_z = tf.nn.relu(tf.matmul(h_flat1_z, W_fc1) + b_fc1)

r0 = tf.reshape(h_fc1_0, [-1, 1, 128, 1])
r1 = tf.reshape(h_fc1_1, [-1, 1, 128, 1])
r2 = tf.reshape(h_fc1_2, [-1, 1, 128, 1])
r3 = tf.reshape(h_fc1_3, [-1, 1, 128, 1])
r4 = tf.reshape(h_fc1_4, [-1, 1, 128, 1])
r5 = tf.reshape(h_fc1_5, [-1, 1, 128, 1])
r6 = tf.reshape(h_fc1_6, [-1, 1, 128, 1])
r7 = tf.reshape(h_fc1_7, [-1, 1, 128, 1])
r8 = tf.reshape(h_fc1_8, [-1, 1, 128, 1])
r9 = tf.reshape(h_fc1_9, [-1, 1, 128, 1])
r10 = tf.reshape(h_fc1_10, [-1, 1, 128, 1])
r11 = tf.reshape(h_fc1_11, [-1, 1, 128, 1])
r12 = tf.reshape(h_fc1_12, [-1, 1, 128, 1])
r13 = tf.reshape(h_fc1_13, [-1, 1, 128, 1])
r14 = tf.reshape(h_fc1_14, [-1, 1, 128, 1])
r15 = tf.reshape(h_fc1_15, [-1, 1, 128, 1])
r16 = tf.reshape(h_fc1_16, [-1, 1, 128, 1])
rz = tf.reshape(h_fc1_z, [-1, 1, 128, 1])

doc_matrix_0 = tf.concat([r0, r1], 1)
doc_matrix_1 = tf.concat([doc_matrix_0, r2], 1)
doc_matrix_2 = tf.concat([doc_matrix_1, r3], 1)
doc_matrix_3 = tf.concat([doc_matrix_2, r4], 1)
doc_matrix_4 = tf.concat([doc_matrix_3, r5], 1)
doc_matrix_5 = tf.concat([doc_matrix_4, r6], 1)
doc_matrix_6 = tf.concat([doc_matrix_5, r7], 1)
doc_matrix_7 = tf.concat([doc_matrix_6, r8], 1)
doc_matrix_8 = tf.concat([doc_matrix_7, r9], 1)
doc_matrix_9 = tf.concat([doc_matrix_8, r10], 1)
doc_matrix_10 = tf.concat([doc_matrix_9, r11], 1)
doc_matrix_11 = tf.concat([doc_matrix_10, r12], 1)
doc_matrix_12 = tf.concat([doc_matrix_11, r13], 1)
doc_matrix_13 = tf.concat([doc_matrix_12, r14], 1)
doc_matrix_14 = tf.concat([doc_matrix_13, r15], 1)
doc_matrix = tf.concat([doc_matrix_14, r16], 1)

# 畳み込み層2
W_conv5 = weight_variable([3, 3, 1, 128])
b_conv5 = bias_variable([128])

W_conv6 = weight_variable([3, 3, 128, 128])
b_conv6 = bias_variable([128])

W_conv7 = weight_variable([3, 3, 128, 128])
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

doc_matrix0_0 = tf.concat([rz, r1], 1)
doc_matrix0_1 = tf.concat([doc_matrix0_0, r2], 1)
doc_matrix0_2 = tf.concat([doc_matrix0_1, r3], 1)
doc_matrix0_3 = tf.concat([doc_matrix0_2, r4], 1)
doc_matrix0_4 = tf.concat([doc_matrix0_3, r5], 1)
doc_matrix0_5 = tf.concat([doc_matrix0_4, r6], 1)
doc_matrix0_6 = tf.concat([doc_matrix0_5, r7], 1)
doc_matrix0_7 = tf.concat([doc_matrix0_6, r8], 1)
doc_matrix0_8 = tf.concat([doc_matrix0_7, r9], 1)
doc_matrix0_9 = tf.concat([doc_matrix0_8, r10], 1)
doc_matrix0_10 = tf.concat([doc_matrix0_9, r11], 1)
doc_matrix0_11 = tf.concat([doc_matrix0_10, r12], 1)
doc_matrix0_12 = tf.concat([doc_matrix0_11, r13], 1)
doc_matrix0_13 = tf.concat([doc_matrix0_12, r14], 1)
doc_matrix0_14 = tf.concat([doc_matrix0_13, r15], 1)
doc_matrix0 = tf.concat([doc_matrix0_14, r16], 1)

doc_matrix1_0 = tf.concat([r0, rz], 1)
doc_matrix1_1 = tf.concat([doc_matrix1_0, r2], 1)
doc_matrix1_2 = tf.concat([doc_matrix1_1, r3], 1)
doc_matrix1_3 = tf.concat([doc_matrix1_2, r4], 1)
doc_matrix1_4 = tf.concat([doc_matrix1_3, r5], 1)
doc_matrix1_5 = tf.concat([doc_matrix1_4, r6], 1)
doc_matrix1_6 = tf.concat([doc_matrix1_5, r7], 1)
doc_matrix1_7 = tf.concat([doc_matrix1_6, r8], 1)
doc_matrix1_8 = tf.concat([doc_matrix1_7, r9], 1)
doc_matrix1_9 = tf.concat([doc_matrix1_8, r10], 1)
doc_matrix1_10 = tf.concat([doc_matrix1_9, r11], 1)
doc_matrix1_11 = tf.concat([doc_matrix1_10, r12], 1)
doc_matrix1_12 = tf.concat([doc_matrix1_11, r13], 1)
doc_matrix1_13 = tf.concat([doc_matrix1_12, r14], 1)
doc_matrix1_14 = tf.concat([doc_matrix1_13, r15], 1)
doc_matrix1 = tf.concat([doc_matrix1_14, r16], 1)

doc_matrix2_0 = tf.concat([r0, r1], 1)
doc_matrix2_1 = tf.concat([doc_matrix2_0, rz], 1)
doc_matrix2_2 = tf.concat([doc_matrix2_1, r3], 1)
doc_matrix2_3 = tf.concat([doc_matrix2_2, r4], 1)
doc_matrix2_4 = tf.concat([doc_matrix2_3, r5], 1)
doc_matrix2_5 = tf.concat([doc_matrix2_4, r6], 1)
doc_matrix2_6 = tf.concat([doc_matrix2_5, r7], 1)
doc_matrix2_7 = tf.concat([doc_matrix2_6, r8], 1)
doc_matrix2_8 = tf.concat([doc_matrix2_7, r9], 1)
doc_matrix2_9 = tf.concat([doc_matrix2_8, r10], 1)
doc_matrix2_10 = tf.concat([doc_matrix2_9, r11], 1)
doc_matrix2_11 = tf.concat([doc_matrix2_10, r12], 1)
doc_matrix2_12 = tf.concat([doc_matrix2_11, r13], 1)
doc_matrix2_13 = tf.concat([doc_matrix2_12, r14], 1)
doc_matrix2_14 = tf.concat([doc_matrix2_13, r15], 1)
doc_matrix2 = tf.concat([doc_matrix2_14, r16], 1)

doc_matrix3_0 = tf.concat([r0, r1], 1)
doc_matrix3_1 = tf.concat([doc_matrix3_0, r2], 1)
doc_matrix3_2 = tf.concat([doc_matrix3_1, rz], 1)
doc_matrix3_3 = tf.concat([doc_matrix3_2, r4], 1)
doc_matrix3_4 = tf.concat([doc_matrix3_3, r5], 1)
doc_matrix3_5 = tf.concat([doc_matrix3_4, r6], 1)
doc_matrix3_6 = tf.concat([doc_matrix3_5, r7], 1)
doc_matrix3_7 = tf.concat([doc_matrix3_6, r8], 1)
doc_matrix3_8 = tf.concat([doc_matrix3_7, r9], 1)
doc_matrix3_9 = tf.concat([doc_matrix3_8, r10], 1)
doc_matrix3_10 = tf.concat([doc_matrix3_9, r11], 1)
doc_matrix3_11 = tf.concat([doc_matrix3_10, r12], 1)
doc_matrix3_12 = tf.concat([doc_matrix3_11, r13], 1)
doc_matrix3_13 = tf.concat([doc_matrix3_12, r14], 1)
doc_matrix3_14 = tf.concat([doc_matrix3_13, r15], 1)
doc_matrix3 = tf.concat([doc_matrix3_14, r16], 1)

doc_matrix4_0 = tf.concat([r0, r1], 1)
doc_matrix4_1 = tf.concat([doc_matrix4_0, r2], 1)
doc_matrix4_2 = tf.concat([doc_matrix4_1, r3], 1)
doc_matrix4_3 = tf.concat([doc_matrix4_2, rz], 1)
doc_matrix4_4 = tf.concat([doc_matrix4_3, r5], 1)
doc_matrix4_5 = tf.concat([doc_matrix4_4, r6], 1)
doc_matrix4_6 = tf.concat([doc_matrix4_5, r7], 1)
doc_matrix4_7 = tf.concat([doc_matrix4_6, r8], 1)
doc_matrix4_8 = tf.concat([doc_matrix4_7, r9], 1)
doc_matrix4_9 = tf.concat([doc_matrix4_8, r10], 1)
doc_matrix4_10 = tf.concat([doc_matrix4_9, r11], 1)
doc_matrix4_11 = tf.concat([doc_matrix4_10, r12], 1)
doc_matrix4_12 = tf.concat([doc_matrix4_11, r13], 1)
doc_matrix4_13 = tf.concat([doc_matrix4_12, r14], 1)
doc_matrix4_14 = tf.concat([doc_matrix4_13, r15], 1)
doc_matrix4 = tf.concat([doc_matrix4_14, r16], 1)

doc_matrix5_0 = tf.concat([r0, r1], 1)
doc_matrix5_1 = tf.concat([doc_matrix5_0, r2], 1)
doc_matrix5_2 = tf.concat([doc_matrix5_1, r3], 1)
doc_matrix5_3 = tf.concat([doc_matrix5_2, r4], 1)
doc_matrix5_4 = tf.concat([doc_matrix5_3, rz], 1)
doc_matrix5_5 = tf.concat([doc_matrix5_4, r6], 1)
doc_matrix5_6 = tf.concat([doc_matrix5_5, r7], 1)
doc_matrix5_7 = tf.concat([doc_matrix5_6, r8], 1)
doc_matrix5_8 = tf.concat([doc_matrix5_7, r9], 1)
doc_matrix5_9 = tf.concat([doc_matrix5_8, r10], 1)
doc_matrix5_10 = tf.concat([doc_matrix5_9, r11], 1)
doc_matrix5_11 = tf.concat([doc_matrix5_10, r12], 1)
doc_matrix5_12 = tf.concat([doc_matrix5_11, r13], 1)
doc_matrix5_13 = tf.concat([doc_matrix5_12, r14], 1)
doc_matrix5_14 = tf.concat([doc_matrix5_13, r15], 1)
doc_matrix5 = tf.concat([doc_matrix5_14, r16], 1)

doc_matrix6_0 = tf.concat([r0, r1], 1)
doc_matrix6_1 = tf.concat([doc_matrix6_0, r2], 1)
doc_matrix6_2 = tf.concat([doc_matrix6_1, r3], 1)
doc_matrix6_3 = tf.concat([doc_matrix6_2, r4], 1)
doc_matrix6_4 = tf.concat([doc_matrix6_3, r5], 1)
doc_matrix6_5 = tf.concat([doc_matrix6_4, rz], 1)
doc_matrix6_6 = tf.concat([doc_matrix6_5, r7], 1)
doc_matrix6_7 = tf.concat([doc_matrix6_6, r8], 1)
doc_matrix6_8 = tf.concat([doc_matrix6_7, r9], 1)
doc_matrix6_9 = tf.concat([doc_matrix6_8, r10], 1)
doc_matrix6_10 = tf.concat([doc_matrix6_9, r11], 1)
doc_matrix6_11 = tf.concat([doc_matrix6_10, r12], 1)
doc_matrix6_12 = tf.concat([doc_matrix6_11, r13], 1)
doc_matrix6_13 = tf.concat([doc_matrix6_12, r14], 1)
doc_matrix6_14 = tf.concat([doc_matrix6_13, r15], 1)
doc_matrix6 = tf.concat([doc_matrix6_14, r16], 1)

doc_matrix7_0 = tf.concat([r0, r1], 1)
doc_matrix7_1 = tf.concat([doc_matrix7_0, r2], 1)
doc_matrix7_2 = tf.concat([doc_matrix7_1, r3], 1)
doc_matrix7_3 = tf.concat([doc_matrix7_2, r4], 1)
doc_matrix7_4 = tf.concat([doc_matrix7_3, r5], 1)
doc_matrix7_5 = tf.concat([doc_matrix7_4, r6], 1)
doc_matrix7_6 = tf.concat([doc_matrix7_5, rz], 1)
doc_matrix7_7 = tf.concat([doc_matrix7_6, r8], 1)
doc_matrix7_8 = tf.concat([doc_matrix7_7, r9], 1)
doc_matrix7_9 = tf.concat([doc_matrix7_8, r10], 1)
doc_matrix7_10 = tf.concat([doc_matrix7_9, r11], 1)
doc_matrix7_11 = tf.concat([doc_matrix7_10, r12], 1)
doc_matrix7_12 = tf.concat([doc_matrix7_11, r13], 1)
doc_matrix7_13 = tf.concat([doc_matrix7_12, r14], 1)
doc_matrix7_14 = tf.concat([doc_matrix7_13, r15], 1)
doc_matrix7 = tf.concat([doc_matrix7_14, r16], 1)

doc_matrix8_0 = tf.concat([r0, r1], 1)
doc_matrix8_1 = tf.concat([doc_matrix8_0, r2], 1)
doc_matrix8_2 = tf.concat([doc_matrix8_1, r3], 1)
doc_matrix8_3 = tf.concat([doc_matrix8_2, r4], 1)
doc_matrix8_4 = tf.concat([doc_matrix8_3, r5], 1)
doc_matrix8_5 = tf.concat([doc_matrix8_4, r6], 1)
doc_matrix8_6 = tf.concat([doc_matrix8_5, r7], 1)
doc_matrix8_7 = tf.concat([doc_matrix8_6, rz], 1)
doc_matrix8_8 = tf.concat([doc_matrix8_7, r9], 1)
doc_matrix8_9 = tf.concat([doc_matrix8_8, r10], 1)
doc_matrix8_10 = tf.concat([doc_matrix8_9, r11], 1)
doc_matrix8_11 = tf.concat([doc_matrix8_10, r12], 1)
doc_matrix8_12 = tf.concat([doc_matrix8_11, r13], 1)
doc_matrix8_13 = tf.concat([doc_matrix8_12, r14], 1)
doc_matrix8_14 = tf.concat([doc_matrix8_13, r15], 1)
doc_matrix8 = tf.concat([doc_matrix8_14, r16], 1)

doc_matrix9_0 = tf.concat([r0, r1], 1)
doc_matrix9_1 = tf.concat([doc_matrix9_0, r2], 1)
doc_matrix9_2 = tf.concat([doc_matrix9_1, r3], 1)
doc_matrix9_3 = tf.concat([doc_matrix9_2, r4], 1)
doc_matrix9_4 = tf.concat([doc_matrix9_3, r5], 1)
doc_matrix9_5 = tf.concat([doc_matrix9_4, r6], 1)
doc_matrix9_6 = tf.concat([doc_matrix9_5, r7], 1)
doc_matrix9_7 = tf.concat([doc_matrix9_6, r8], 1)
doc_matrix9_8 = tf.concat([doc_matrix9_7, rz], 1)
doc_matrix9_9 = tf.concat([doc_matrix9_8, r10], 1)
doc_matrix9_10 = tf.concat([doc_matrix9_9, r11], 1)
doc_matrix9_11 = tf.concat([doc_matrix9_10, r12], 1)
doc_matrix9_12 = tf.concat([doc_matrix9_11, r13], 1)
doc_matrix9_13 = tf.concat([doc_matrix9_12, r14], 1)
doc_matrix9_14 = tf.concat([doc_matrix9_13, r15], 1)
doc_matrix9 = tf.concat([doc_matrix9_14, r16], 1)

doc_matrix10_0 = tf.concat([r0, r1], 1)
doc_matrix10_1 = tf.concat([doc_matrix10_0, r2], 1)
doc_matrix10_2 = tf.concat([doc_matrix10_1, r3], 1)
doc_matrix10_3 = tf.concat([doc_matrix10_2, r4], 1)
doc_matrix10_4 = tf.concat([doc_matrix10_3, r5], 1)
doc_matrix10_5 = tf.concat([doc_matrix10_4, r6], 1)
doc_matrix10_6 = tf.concat([doc_matrix10_5, r7], 1)
doc_matrix10_7 = tf.concat([doc_matrix10_6, r8], 1)
doc_matrix10_8 = tf.concat([doc_matrix10_7, r9], 1)
doc_matrix10_9 = tf.concat([doc_matrix10_8, rz], 1)
doc_matrix10_10 = tf.concat([doc_matrix10_9, r11], 1)
doc_matrix10_11 = tf.concat([doc_matrix10_10, r12], 1)
doc_matrix10_12 = tf.concat([doc_matrix10_11, r13], 1)
doc_matrix10_13 = tf.concat([doc_matrix10_12, r14], 1)
doc_matrix10_14 = tf.concat([doc_matrix10_13, r15], 1)
doc_matrix10 = tf.concat([doc_matrix10_14, r16], 1)

doc_matrix11_0 = tf.concat([r0, r1], 1)
doc_matrix11_1 = tf.concat([doc_matrix11_0, r2], 1)
doc_matrix11_2 = tf.concat([doc_matrix11_1, r3], 1)
doc_matrix11_3 = tf.concat([doc_matrix11_2, r4], 1)
doc_matrix11_4 = tf.concat([doc_matrix11_3, r5], 1)
doc_matrix11_5 = tf.concat([doc_matrix11_4, r6], 1)
doc_matrix11_6 = tf.concat([doc_matrix11_5, r7], 1)
doc_matrix11_7 = tf.concat([doc_matrix11_6, r8], 1)
doc_matrix11_8 = tf.concat([doc_matrix11_7, r9], 1)
doc_matrix11_9 = tf.concat([doc_matrix11_8, r10], 1)
doc_matrix11_10 = tf.concat([doc_matrix11_9, rz], 1)
doc_matrix11_11 = tf.concat([doc_matrix11_10, r12], 1)
doc_matrix11_12 = tf.concat([doc_matrix11_11, r13], 1)
doc_matrix11_13 = tf.concat([doc_matrix11_12, r14], 1)
doc_matrix11_14 = tf.concat([doc_matrix11_13, r15], 1)
doc_matrix11 = tf.concat([doc_matrix11_14, r16], 1)

doc_matrix12_0 = tf.concat([r0, r1], 1)
doc_matrix12_1 = tf.concat([doc_matrix12_0, r2], 1)
doc_matrix12_2 = tf.concat([doc_matrix12_1, r3], 1)
doc_matrix12_3 = tf.concat([doc_matrix12_2, r4], 1)
doc_matrix12_4 = tf.concat([doc_matrix12_3, r5], 1)
doc_matrix12_5 = tf.concat([doc_matrix12_4, r6], 1)
doc_matrix12_6 = tf.concat([doc_matrix12_5, r7], 1)
doc_matrix12_7 = tf.concat([doc_matrix12_6, r8], 1)
doc_matrix12_8 = tf.concat([doc_matrix12_7, r9], 1)
doc_matrix12_9 = tf.concat([doc_matrix12_8, r10], 1)
doc_matrix12_10 = tf.concat([doc_matrix12_9, r11], 1)
doc_matrix12_11 = tf.concat([doc_matrix12_10, rz], 1)
doc_matrix12_12 = tf.concat([doc_matrix12_11, r13], 1)
doc_matrix12_13 = tf.concat([doc_matrix12_12, r14], 1)
doc_matrix12_14 = tf.concat([doc_matrix12_13, r15], 1)
doc_matrix12 = tf.concat([doc_matrix12_14, r16], 1)

doc_matrix13_0 = tf.concat([r0, r1], 1)
doc_matrix13_1 = tf.concat([doc_matrix13_0, r2], 1)
doc_matrix13_2 = tf.concat([doc_matrix13_1, r3], 1)
doc_matrix13_3 = tf.concat([doc_matrix13_2, r4], 1)
doc_matrix13_4 = tf.concat([doc_matrix13_3, r5], 1)
doc_matrix13_5 = tf.concat([doc_matrix13_4, r6], 1)
doc_matrix13_6 = tf.concat([doc_matrix13_5, r7], 1)
doc_matrix13_7 = tf.concat([doc_matrix13_6, r8], 1)
doc_matrix13_8 = tf.concat([doc_matrix13_7, r9], 1)
doc_matrix13_9 = tf.concat([doc_matrix13_8, r10], 1)
doc_matrix13_10 = tf.concat([doc_matrix13_9, r11], 1)
doc_matrix13_11 = tf.concat([doc_matrix13_10, r12], 1)
doc_matrix13_12 = tf.concat([doc_matrix13_11, rz], 1)
doc_matrix13_13 = tf.concat([doc_matrix13_12, r14], 1)
doc_matrix13_14 = tf.concat([doc_matrix13_13, r15], 1)
doc_matrix13 = tf.concat([doc_matrix13_14, r16], 1)

doc_matrix14_0 = tf.concat([r0, r1], 1)
doc_matrix14_1 = tf.concat([doc_matrix14_0, r2], 1)
doc_matrix14_2 = tf.concat([doc_matrix14_1, r3], 1)
doc_matrix14_3 = tf.concat([doc_matrix14_2, r4], 1)
doc_matrix14_4 = tf.concat([doc_matrix14_3, r5], 1)
doc_matrix14_5 = tf.concat([doc_matrix14_4, r6], 1)
doc_matrix14_6 = tf.concat([doc_matrix14_5, r7], 1)
doc_matrix14_7 = tf.concat([doc_matrix14_6, r8], 1)
doc_matrix14_8 = tf.concat([doc_matrix14_7, r9], 1)
doc_matrix14_9 = tf.concat([doc_matrix14_8, r10], 1)
doc_matrix14_10 = tf.concat([doc_matrix14_9, r11], 1)
doc_matrix14_11 = tf.concat([doc_matrix14_10, r12], 1)
doc_matrix14_12 = tf.concat([doc_matrix14_11, r13], 1)
doc_matrix14_13 = tf.concat([doc_matrix14_12, rz], 1)
doc_matrix14_14 = tf.concat([doc_matrix14_13, r15], 1)
doc_matrix14 = tf.concat([doc_matrix14_14, r16], 1)

doc_matrix15_0 = tf.concat([r0, r1], 1)
doc_matrix15_1 = tf.concat([doc_matrix15_0, r2], 1)
doc_matrix15_2 = tf.concat([doc_matrix15_1, r3], 1)
doc_matrix15_3 = tf.concat([doc_matrix15_2, r4], 1)
doc_matrix15_4 = tf.concat([doc_matrix15_3, r5], 1)
doc_matrix15_5 = tf.concat([doc_matrix15_4, r6], 1)
doc_matrix15_6 = tf.concat([doc_matrix15_5, r7], 1)
doc_matrix15_7 = tf.concat([doc_matrix15_6, r8], 1)
doc_matrix15_8 = tf.concat([doc_matrix15_7, r9], 1)
doc_matrix15_9 = tf.concat([doc_matrix15_8, r10], 1)
doc_matrix15_10 = tf.concat([doc_matrix15_9, r11], 1)
doc_matrix15_11 = tf.concat([doc_matrix15_10, r12], 1)
doc_matrix15_12 = tf.concat([doc_matrix15_11, r13], 1)
doc_matrix15_13 = tf.concat([doc_matrix15_12, r14], 1)
doc_matrix15_14 = tf.concat([doc_matrix15_13, rz], 1)
doc_matrix15 = tf.concat([doc_matrix15_14, r16], 1)

doc_matrix16_0 = tf.concat([r0, r1], 1)
doc_matrix16_1 = tf.concat([doc_matrix16_0, r2], 1)
doc_matrix16_2 = tf.concat([doc_matrix16_1, r3], 1)
doc_matrix16_3 = tf.concat([doc_matrix16_2, r4], 1)
doc_matrix16_4 = tf.concat([doc_matrix16_3, r5], 1)
doc_matrix16_5 = tf.concat([doc_matrix16_4, r6], 1)
doc_matrix16_6 = tf.concat([doc_matrix16_5, r7], 1)
doc_matrix16_7 = tf.concat([doc_matrix16_6, r8], 1)
doc_matrix16_8 = tf.concat([doc_matrix16_7, r9], 1)
doc_matrix16_9 = tf.concat([doc_matrix16_8, r10], 1)
doc_matrix16_10 = tf.concat([doc_matrix16_9, r11], 1)
doc_matrix16_11 = tf.concat([doc_matrix16_10, r12], 1)
doc_matrix16_12 = tf.concat([doc_matrix16_11, r13], 1)
doc_matrix16_13 = tf.concat([doc_matrix16_12, r14], 1)
doc_matrix16_14 = tf.concat([doc_matrix16_13, r15], 1)
doc_matrix16 = tf.concat([doc_matrix16_14, rz], 1)

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

sentence_num_train = np.zeros((1))
sentence_num_train[0] = -1
sentence_num_train = np.array(sentence_num_train, dtype=np.int32)
out = y_out.eval(session=sess, feed_dict={x: batch, y_: output, sentence_num: sentence_num_train})
output = out

loss_list = []
for i in range(max_sentence_size):
    sentence_num_train[0] = int(i)
    loss = cross_entropy.eval(session=sess, feed_dict={x: batch, y_: output, sentence_num: sentence_num_train})
    loss_list.append(loss)
    aaa = sentence_num.eval(session=sess, feed_dict={x: batch, y_: output, sentence_num: sentence_num_train})
sess.close()