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
from keras.layers import Dense, Activation,Dropout
from sklearn.model_selection import train_test_split
from progressbar import ProgressBar

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
        files = [file for file in os.listdir(path + '/' + label) if file.endswith('.jpeg')]
        for file in files:
            tmp = []
            img = cv2.imread(path + '/' + label + '/' + file)
            color = ('b','g','r')
            for i,col in enumerate(color):
                histr = cv2.calcHist([img],[i],None,[256],[0,256])
                tmp.append(histr)
                #img = cv2.resize(img,(200,200))
                #tmp.append(img.ravel())
            X.append(np.transpose(np.vstack(tmp)))
            
            y.append(int(label)-1)
    return np.vstack(X),y   

def genImages(path):
    datagen = ImageDataGenerator(
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            rescale=1./255,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            fill_mode='nearest')
    
    for label in os.listdir(path):
        files = [file for file in os.listdir(path + '/' + label) if file.endswith('.jpg')]
        for file in files:
            img = load_img(path + '/' + label + '/' + file)  # this is a PIL image
            x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
            x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)
            
            # the .flow() command below generates batches of randomly transformed images
            # and saves the results to the `preview/` directory
            i = 0
            for batch in datagen.flow(x, batch_size=1,
                                      save_to_dir=path + '/' + label, save_prefix=label, save_format='jpeg'):
                i += 1
                if i > 20:
                    break  # otherwise the generator would loop indefinitely 

pbar = ProgressBar()
scores = []
for i in pbar(range(10)):
    X_train,y_train = load(path)
    X_train,X_test,y_train,y_test = train_test_split(X_train,y_train,test_size=0.1)
    
    model = Sequential([
        Dense(1000, input_shape=(X_train.shape[1],)),
        Activation('tanh'),
        Dropout(0.0),
        Dense(len(np.unique(y_train))),
        Activation('softmax'),
    ])
        
    model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
    
    model.fit(X_train, y_train, epochs=10, batch_size=100,shuffle=True,verbose=1)
    out = model.predict_classes(X_test)
    score = 0
    for o,yi in zip(out,y_test):
        if o == yi:
            score+=1
    scores.append(score/len(out)*100)
    print('\n' + str(scores[-1]))
        
    