# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 12:14:45 2018

@author: ohaddoron
"""


import os

def load_labels(settings,params,purpose='Evaluation data'):
    path = settings.path2data + '/' + purpose
    # find the annotations file in the list of files
    for file in os.listdir(path):
        if file.endswith('.txt'):
            fname = file
    # save locations into dictionary
    locs = {}
    with open(settings.path2data + '/' + purpose + '/' + fname) as f:
        for line in f:
            name = line.split(':')[0]
            locs[name] = line.split(':')[1]
    return locs
        
    
    