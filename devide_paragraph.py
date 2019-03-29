# -*- coding: utf-8 -*-

import pickle

paper_org = pickle.load(open("energy_paper_2018_dict.pickle", "rb"))

not_figure_dict = {}
figure_dict = {}

for key in paper_org:
    para_list = paper_org[key]
    not_figure_list = []
    figure_list = []
    
    for para in para_list:
        if para.find("（第") != -1:
            figure_list.append(para)
        else:
            not_figure_list.append(para)
    not_figure_dict[key] = not_figure_list
    figure_dict[key] = figure_list
    
pickle.dump(not_figure_dict, open("not_figure_dict.pickle", "wb"))
pickle.dump(figure_dict, open("figure_dict.pickle", "wb"))