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
            imName = line.split(':')[0]
            bus = os.path.join(settings.busOriginalDir, imName)
            locs[imName] = []
            for loc in line.split(':')[1].split('],'):
                loc = [int(i) for i in loc.replace('[','').replace(']','').split(',')]
                locs[imName].append(loc)
    return locs
def load_colors(settings):
    # save colors of buses redt into dictionary
    colors = {}
    with open(settings.foreground_colors, 'r') as colorFileGT:
         for line in colorFileGT:
            imName = line.split(':')[0]
            color = line.split(':')[1].replace('\n', '')
            colors[imName] = color
    return colors

def create_bus_rect_img(settings):
    locs=load_labels(settings)
    for name in locs:
        # bus = os.path.join(settings.busOriginalDir, name)
        # img=cv2.imread(bus)
        for i,rect in enumerate(locs[name]):
            # temp=img[rect[1]:rect[1]+rect[3],rect[0]:rect[0]+rect[2],:]
            # cv2.imwrite('busesRect/'+name.replace('.JPG','_' + str(i) + '.jpg'),temp)
            f=open('busesRect/bus_color.txt', "a+")
            f.write(name.replace('.JPG','_' + str(i) + '.jpg:') +str(rect[4])+ '\n')

def resize_background(settings,params):
    background_names = os.listdir(settings.path2background)
    for background_image in background_names:
        image = cv2.imread(settings.path2background + '/'
                                     + background_image)
        image_res = cv2.resize(image, (params.image_height, params.image_width))
        cv2.imwrite(settings.path2background + '/resized/' + background_image,image_res)
def mirror_images(settings, params):
    locs=load_labels(settings)
    for name in locs:
        bus = os.path.join(settings.busOriginalDir, name)
        img=cv2.imread(bus)
        img_flip =cv2.flip(img,1,settings.path2background + '/mirrored/' +name.replace('.JPG','_mirror.jpg'))
        for i,rect in enumerate(locs[name]):
            temp=img[rect[1]:rect[1]+rect[3],rect[0]:rect[0]+rect[2],:]
            # cv2.imwrite(settings.path2background + '/mirrored/' +name.replace('.JPG','_' + str(i) + '.jpg'),temp)
            f=open(settings.path2background + '/mirrored'+'/bus_color.txt', "a+")
            f.write(name.replace('.JPG','_' + str(i) + '.jpg:') +str(rect[4])+ '\n')



def crop_background(settings,params):
    background_names = os.listdir(settings.path2background + '/resized')
    for background_image in background_names:
        for i in range(np.random.randint(low=2,high=5)):
            image = cv2.imread(settings.path2background + '/resized/'
                                         + background_image)
            x=np.random.randint(low=0,high=params.image_height-1000)
            dx=np.random.randint(low=500,high=params.image_height-x)
            y=np.random.randint(low=0,high=params.image_width-1000)
            dy=np.random.randint(low=500,high=params.image_width-y)
            image=image[x:x+dx, y:y+dy,:]
            image_res = cv2.resize(image, (params.image_height, params.image_width))
            cv2.imwrite(settings.path2background + '/cropped/'+str(i)+'_' + background_image,image_res)


settings,params=settings_params.load()
mirror_images(settings,params)

