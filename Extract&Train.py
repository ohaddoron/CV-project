# -*- coding: utf-8 -*-
"""
Created on Tue May 15 14:20:57 2018

@author: ohaddoron
"""

import os
import shutil
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation

path = r'C:\Users\ohaddoron\Documents\CV Project\busesRect'
annotations = path + '/bus_color.txt'

def move_files(path,annotations):
    with open(annotations ,'r') as f:
        for line in f:
            name = line.split(':')[0]
            label = line.split(':')[1].replace('\n','')
            if not os.path.isdir(path + '/' + label):
                os.mkdir(path + '/' + label)
            shutil.move(path + '/' + name,path + '/' + label + '/' + name)
            
def load(path):
    X = []
    y = []
    for label in os.listdir(path):
        files = [file for file in os.listdir(path + '/' + label) if file.endswith('.jpg')]
        for file in files:
            tmp = []
            img = cv2.imread(path + '/' + label + '/' + file)
            color = ('b','g','r')
            for i,col in enumerate(color):
                histr = cv2.calcHist([img],[i],None,[256],[0,256])
                tmp.append(histr)
            X.append(np.transpose(np.vstack(tmp)))
            y.append(int(label))
    return np.vstack(X),y    

X_train,y_train = load(path)

model = Sequential([
    Dense(200, input_shape=(X_train.shape[1],)),
    Activation('tanh'),
    Dense(len(np.unique(y_train))),
    Activation('softmax'),
])
    
model.compile(optimizer='adam',
          loss='categorical_crossentropy',
          metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10, batch_size=32)