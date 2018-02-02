# -*- coding: utf-8 -*-
"""
Created on Fri Feb 02 15:42:25 2018

@author: ndoannguyen
"""

import os

# Exercise 1
from bs4 import BeautifulSoup

def prepareDataFolder(data_folder):
    # Exercise 2
    # TODO
    if (os.path.isfile(data_folder)):
        print("A file named '" + data_folder + "' already existed." )
    elif (os.path.isdir(data_folder)):
        print("Folder '" + data_folder + "' already existed." )
    else:
        os.mkdir(data_folder)