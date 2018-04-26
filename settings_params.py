# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 11:54:26 2018

@author: ohaddoron
"""

class settings:
    # settings to be used throughout the project    
    def __init__(self):
        self.path2data = '../data'
        self.path2results = '../results'
        self.path2foreground = '../data/foreground'
        self.path2background = '../data/background'
        self.path2dev = '../data/dev'
    
class params:
    # parameters to be used throughout the project
    def __init__(self):
        self.seed = 1
        self.num_training_images = 10000
        self.max_objects = 5

    
def load():
    return settings(),params()