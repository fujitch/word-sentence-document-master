# -*- coding: utf-8 -*-
"""
単語の出現回数の上位n個のdictを作る
"""

import pickle
import MeCab

max_count = 10

m_ochasen = MeCab.Tagger(r'-Ochasen -d C:\Users\hori\workspace\encoder-decoder-sentence-chainer-master\mecab-ipadic-neologd')
part_chapter_dict = pickle.load(open("energy_paper_2018_dict.pickle", "rb"))

frequent_dict = {}
for key in part_chapter_dict:
    part_chapter = part_chapter_dict[key]
    for paragraph in part_chapter:
        node = m_ochasen.parseToNode(paragraph)
        while node:
            fields = node.feature.split(",")
            if fields[0] == '名詞' or fields[0] == '動詞' or fields[0] == '形容詞':
                if node.surface in frequent_dict:
                    frequent_dict[node.surface] = frequent_dict[node.surface] + 1
                else:
                    frequent_dict[node.surface] = 1
            node = node.next

frequent_limit_dict = {}
counter = 0
for k, v in sorted(frequent_dict.items(), key=lambda x: -x[1]):
    frequent_limit_dict[k] = v
    counter += 1
    if counter == 10:
        break
pickle.dump(frequent_limit_dict, open("frequent_dict.pickle", "wb"))