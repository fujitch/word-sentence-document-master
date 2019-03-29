# -*- coding: utf-8 -*-

import MySQLdb

connection = MySQLdb.connect(host='localhost', user='root', passwd = 'root', db='energy_white_paper', charset='utf8')
cursor = connection.cursor()

cursor.execute("select*from paragraph")
all_data = cursor.fetchall()

dataset_dict = {}

part = 999
chapter = 999
section = 999
text_list = []
text = ""

for data in all_data:
    if str(data[2]) != str(part):
        if text != "":
            text_list.append(text)
            text = ""
            dataset_dict[str(part) + "_" + str(chapter)] = text_list
        part = data[2]
    if str(data[3]) != str(chapter):
        if text != "":
            text_list.append(text)
            text = ""
            dataset_dict[str(part) + "_" + str(chapter)] = text_list
        chapter = data[3]
    if str(data[4]) != str(section):
        if text != "":
            text_list.append(text)
            text = ""
            dataset_dict[str(part) + "_" + str(chapter)] = text_list
        section = data[4]
    text = text + data[7]