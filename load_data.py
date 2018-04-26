# -*- coding: utf-8 -*-
"""
Created on Thu Mar 15 12:14:45 2018

@author: ohaddoron
"""

import cv2
import os
import settings_params
import numpy as np

def load_labels(settings):
    # save locations into dictionary
    locs = {}
    with open(settings.GT_annotations, 'r') as annFileGT:
         for line in annFileGT:
            colors = []
            imName = line.split(':')[0]
            bus = os.path.join(settings.busOriginalDir, imName)
            locs[imName] = []
            for loc in line.split(':')[1].split('],'):
                loc = [int(i) for i in loc.replace('[','').replace(']','').split(',')]
                locs[imName].append(loc)
    return locs

def create_bus_rect_img(settings):
    locs=load_labels(settings)

    for name in locs:
        bus = os.path.join(settings.busOriginalDir, name)
        img=cv2.imread(bus)
        for i,rect in enumerate(locs[name]):
            temp=img[rect[1]:rect[1]+rect[3],rect[0]:rect[0]+rect[2],:]
            cv2.imwrite('busesRect/'+name.replace('.JPG','_' + str(i) + '.jpg'),temp)


def resize_background(setting,params):
    background_names = os.listdir(settings.path2background)
    backgroud_image = cv2.imread(settings.path2background + '/'
                                     + background_names[np.random.randint(low=0,high=len(background_names)-1)])
    backgroud_image_re = cv2.resize(backgroud_image, (params.image_height, params.image_width))


settings,params=settings_params.load()
resize_background(settings,params)

