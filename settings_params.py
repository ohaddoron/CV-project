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
        self.path2foreground = './busesRect'
        self.foreground_colors='./busesRect_color.txt'
        self.path2background = './Background'
        self.path2dev = './dev'
        self.GT_annotations='./annotationsTrain.txt'
        self.busOriginalDir='./busesTrain'
class params:
    # parameters to be used throughout the project
    def __init__(self):
        self.seed = 1
        self.num_training_images = 1000
        self.max_objects = 5
        self.image_height=2736
        self.image_width=3648

    
def load():
    return settings(),params()
