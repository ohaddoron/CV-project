#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu May 24 12:28:11 2018

@author: ohad
"""
import cv2
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from skimage import io
from keras.models import load_model

def evaluate(model1,model2,img):
    img2 = cv2.resize(img,(150,150))
    img2 = img2[np.newaxis,:]
    test_datagen = ImageDataGenerator(
        # preprocessing_function=convert_img,
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)
    
    X = test_datagen.flow(img2)
    
    init_class = np.argmax(model1.predict_generator(X))
    
    if init_class in [2,3]:
        X = extract(img)
        final_class = model2.predict_classes(X) + 2
    else:
        final_class = init_class
    return final_class + 1
        
def extract(img):
    # img/=np.max(img)
    tmp = []
    color = ('r','g','b')
    for i,c in enumerate(color):
        histr = np.histogram(img[:,:,i],bins=256)[0]
        tmp = np.hstack((tmp,histr))
    return np.transpose(np.vstack(tmp))



