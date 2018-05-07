# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 15:32:58 2018

@author: ohaddoron
"""


import cv2
import numpy as np
from PIL import Image
import os
import random
import settings_params
from load_data import *
abc = 1
def genTrainingData(settings,params):
    background_names = os.listdir(settings.path2background+'/cropped')
    foreground_names = os.listdir(settings.path2foreground)
    colors = load_colors(settings)
    # offsets = ()
    for i in range(params.num_training_images):
        background_image = cv2.imread(settings.path2background + '/cropped/'
                                     + background_names[np.random.randint(low=0,high=len(background_names)-1)])
        # offsets += ([],)
        offsets=[]
        if params.max_objects == 1:
            num_objects = 1
        else:
            num_objects = np.random.randint(low=1,high=params.max_objects)
        for j in range(num_objects):
            bus_img = random.choice(foreground_names)
            foreground_image = cv2.imread(settings.path2foreground + '/'
                                          + bus_img)
            background_image,cur_offset = blend_images(foreground_image,background_image)
            cur_offset.append(int(colors[bus_img]))
            # offsets.append(cur_offset + (foreground_image.shape[:2]))

            offsets.append(cur_offset)
        cv2.imwrite(settings.path2dev + '/' + str(i) + '.JPG',background_image)
    # np.save(settings.path2dev +'/' + 'annotations.npy',offsets)
        f=open(settings.path2dev +'/'+'newANN.txt', "a+")
        offset = ','.join(map(str, offsets))
        f.write(str(i) + '.JPG:' + offset +'\n')


def blend_images(foreground, background):
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
    x_offset = np.random.randint(low=0,high=(background.shape[1]-foreground.shape[1])) # horizontal direction
    y_offset = np.random.randint(low=0,high=(background.shape[0]-foreground.shape[0])) # vertical direction
    
    # offset = [x_offset,y_offset,foreground.shape[1]]
    offset = [x_offset,y_offset]

    MASK = 255 * cv2.morphologyEx(alpha, cv2.MORPH_CLOSE, kernel).astype('uint8')
    
    
    background = Image.fromarray(background)
    foreground = Image.fromarray(foreground)
    MASK = Image.fromarray(MASK)

    
    background.paste(foreground, offset, MASK)
    
    return np.array(background),offset
    





settings,params = settings_params.load()
genTrainingData(settings,params)
