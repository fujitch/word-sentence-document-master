# -*- coding: utf-8 -*-

from gensim.models import word2vec
import pickle
import numpy as np
import tensorflow as tf

part_chapter_dict_wakati_train = pickle.load(open("not_figure_dict_wakati.pickle", "rb"))

part_chapter_dict = pickle.load(open("figure_dict.pickle", "rb"))
part_chapter_dict_wakati = pickle.load(open("figure_dict_wakati.pickle", "rb"))

max_sentence_size = 17
max_word_size = 186

dict_keys_list = list(part_chapter_dict_wakati.keys())

all_max = 0.0
all_min = 0.0
for k in part_chapter_dict_wakati_train:
    for sentence_list in part_chapter_dict_wakati_train[k]:
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
y_out_before = tf.matmul(h_fc2, W_fc3) + b_fc3
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

ind = -1
out_dict = {}
loss_dict = {}
sorted_paragraph_dict = {}

for k in part_chapter_dict_wakati:
    sess = tf.Session()
    saver = tf.train.Saver()
    saver.restore(sess, "./model20190210/not_figure_dict_model")
    ind += 1
    para_list = part_chapter_dict_wakati[k]
    if len(para_list) == 0:
        continue
    batch_size = len(para_list)
    batch = np.zeros((batch_size, max_sentence_size * max_word_size * 200))
    batch = np.array(batch, dtype=np.float32)
    output = np.zeros((batch_size, 14))
    output = np.array(output, dtype=np.int32)
    for i in range(batch_size):
        datas = para_list[i]
        batch_before = np.zeros((max_word_size, 200, max_sentence_size))
        for l in range(len(datas)):
            data = datas[l]
            dum_matrix = np.zeros((max_word_size, 200))
            dum_matrix[:data.shape[0], :] = data
            batch_before[:, :, l] = dum_matrix
        batch[i, :] = np.reshape(batch_before, (max_sentence_size * max_word_size * 200))
        output[i, ind] = 1
    
    batch_zeros = np.zeros((batch_size, max_word_size * 200))
    batch_zeros = np.array(batch_zeros, dtype=np.float32)

    out_dict[k] = y_out_before.eval(session=sess, feed_dict={x: batch, y_: output, dum_zeros: batch_zeros})
    loss_dict[k] = cross_entropy.eval(session=sess, feed_dict={x: batch, y_: output, dum_zeros: batch_zeros})
    sort_mat = np.argsort(out_dict[k][:, ind])[::-1]
    para_list = []
    for num in sort_mat:
        para_list.append(part_chapter_dict[k][num])
    sorted_paragraph_dict[k] = para_list
    sess.close()
    