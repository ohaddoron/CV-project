# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 15:32:58 2018

@author: ohaddoron
"""


import cv2
import numpy as np
from PIL import Image
import os
import settings_params
abc = 1
def genTrainingData(settings,params):
    background_names = os.listdir(settings.path2background)
    foreground_names = os.listdir(settings.path2foreground)
    offsets = ()
    for i in range(params.num_training_images):
        backgroud_image = cv2.imread(settings.path2background + '/'
                                     + background_names[np.random.randint(low=0,high=len(background_names)-1)])
        offsets += ([],)
        if params.max_objects == 1:
            num_objects = 1
        else:
            num_objects = np.random.randint(low=1,high=params.max_objects)
        for j in range(num_objects):
            foreground_image = cv2.imread(settings.path2foreground + '/'
                                          + foreground_names[np.random.randint(low=0,high=len(foreground_names)-1)])
            backgroud_image,cur_offset = blend_images(foreground_image,backgroud_image)
            
            offsets[-1].append(cur_offset + (foreground_image.shape[:2]))
        cv2.imwrite(settings.path2dev + '/' + str(i) + '.jpg',backgroud_image)
    np.save(settings.path2dev +'/' + 'annotations.npy',offsets)
        


def blend_images(foreground,background):
    '''
    inputs:
        forground - image to be placed in the front. In this case, of a toy bus
        background - image to be placed in the back. Can be an image of any scene
    Outputs:
        blended images
    Assumptions:
        It is assumed the the foreground image is smaller 
        and is surrounded by a white margin
    '''
    
    alpha = np.zeros(foreground.shape[:2])
    alpha[foreground[:,:,-1] < 255] = 1
    kernel = np.ones((5,5),np.uint8)
    x_offset = np.random.randint(low=0,high=background.shape[1]-foreground.shape[1])
    y_offset = np.random.randint(low=0,high=background.shape[0]-foreground.shape[0])
    
    offset = (x_offset,y_offset)
    MASK = 255 * cv2.morphologyEx(alpha, cv2.MORPH_CLOSE, kernel).astype('uint8')
    
    
    background = Image.fromarray(background)
    foreground = Image.fromarray(foreground)
    MASK = Image.fromarray(MASK)
    
    
    
    
    background.paste(foreground, offset, MASK)
    
    return np.array(background),offset
    

settings,params = settings_params.load()
offsets = genTrainingData(settings,params)
