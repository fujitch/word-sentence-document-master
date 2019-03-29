# -*- coding: utf-8 -*-

from selenium import webdriver
import pickle

years = [2018]

driver = webdriver.Chrome()

for year in years:
    
    page_url = "http://www.enecho.meti.go.jp/about/whitepaper/" + str(year) + "html/"
    
    driver.get(page_url)
    
    links = []
    
    for a_element in driver.find_elements_by_tag_name('a'):
        if a_element.get_attribute('href').find(page_url) != -1 and a_element.get_attribute('href').find(page_url + '#') == -1 and a_element.text.find('はじめに') == -1:
            links.append(a_element.get_attribute('href'))
    links = links[:len(links)-1]
    
    part_chapter_dict = {}
    
    for link in links:
        driver.get(link)
        if link[55:58] in part_chapter_dict:
            part_chapter_list = part_chapter_dict[link[55:58]]
            text = []
            for p_element in driver.find_elements_by_tag_name('p'):
                if p_element.get_attribute('class') == "caption" and p_element.text.find('Adobe Acrobat Reader') == -1 and p_element.text.find('Copyright') == -1 and p_element.text.find('ppt/pptx形式') == -1:
                    text.append(p_element.text)
            part_chapter_list.extend(text)
            part_chapter_dict[link[55:58]] = part_chapter_list
        else:
            part_chapter_list = []
            text = []
            for p_element in driver.find_elements_by_tag_name('p'):
                if p_element.get_attribute('class') == "caption" and p_element.text.find('Adobe Acrobat Reader') == -1 and p_element.text.find('Copyright') == -1 and p_element.text.find('ppt/pptx形式') == -1:
                    text.append(p_element.text)
            part_chapter_list.extend(text)
            part_chapter_dict[link[55:58]] = part_chapter_list
    
    fname = "energy_paper_" + str(year) + "_title.pickle"
    pickle.dump(part_chapter_dict, open(fname, 'wb'))