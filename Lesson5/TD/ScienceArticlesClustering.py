# -*- coding: utf-8 -*-
"""
Created on Tue Feb 06 15:28:46 2018

@author: ndoannguyen
"""

import os, re
import sys, codecs

sys.stdout = codecs.getwriter('utf_8')(sys.stdout)
sys.stdin = codecs.getreader('utf_8')(sys.stdin)

def concatenateDataFiles(raw_data_folder, full_data_folder, all_data_file):
    # Exercise 1
    # TODO
    if not os.path.exists(raw_data_folder):
        os.mkdir(raw_data_folder)
    if not os.path.exists(full_data_folder):
        os.mkdir(full_data_folder)
    raw_file_names = os.listdir(raw_data_folder)
    new_file = open(all_data_file, 'w')
    for raw_file_name in raw_file_names:
        print("Cleaning %s..." % raw_data_folder + "/" + raw_file_name)
        raw_file = open(raw_data_folder + "/" + raw_file_name, 'r')
        s = raw_file.read()
        data = s.split("\n")
        cleaned_data = "\n".join(filter(lambda data: len(data) > 0, data))
        new_file.write(cleaned_data)
        new_file.write("\n")
        raw_file.close()
    new_file.close()

def paragraphToSentences(paragraph):
    # Exercise 2
    # TODO
    text = re.sub(' +', ' ', paragraph)
    text = text.replace(". ",".<stop>").replace("? ","?<stop>").replace("! ","!<stop>")
    sentences = text.split("<stop>")
    return sentences

def sentenceToSegments(sentence):
    # Exercise 3
    # TODO
    text = re.sub(' +', ' ', sentence)
    text = text.replace(", ",",<stop>").replace(": ",":<stop>").replace("; ",";<stop>").replace("- ",";<stop>").replace(") ",")<stop>").replace(" (","<stop>(")
    segments = text.split("<stop>")
    return segments

def clean(segment):
    pattern = re.compile(r'[,;$\:\-\"\'\(\)\/\@\*\&\%\“\.{1,}\?\!\d]')
    text = re.sub(pattern, '', segment)
    text = re.sub(' +', ' ', text)
    return text

def segmentToUnits(segment):
    # Exercise 5
    # TODO
    segment = clean(segment)
    return segment.split(" ")

print len(u'Lưu Tuấn Anh')