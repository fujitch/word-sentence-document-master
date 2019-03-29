# -*- coding: utf-8 -*-

import pickle
import MeCab
import numpy as np
from gensim.models import word2vec

part_chapter_dict = pickle.load(open("not_figure_dict.pickle", "rb"))
batch_size = 9 * len(part_chapter_dict)
m_owakati = MeCab.Tagger(r'-Owakati -d C:\Users\hori\workspace\encoder-decoder-sentence-chainer-master\mecab-ipadic-neologd')
part_chapter_dict_wakati = {}
model = pickle.load(open('word2vec_neologd30.pickle', 'rb'))

for k in part_chapter_dict:
    part_chapter = part_chapter_dict[k]
    wakati_list = []
    for texts in part_chapter:
        text_list = []
        texts = texts.split("。")
        if texts[len(texts)-1] == "":
            texts = texts[:len(texts)-1]
        for text in texts:
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
        wakati_list.append(text_list)
    part_chapter_dict_wakati[k] = wakati_list
pickle.dump(part_chapter_dict_wakati, open('not_figure_dict_wakati.pickle', 'wb'))