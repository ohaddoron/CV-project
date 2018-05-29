# -*- coding: utf-8 -*-
"""
Created on Tue May 15 14:20:57 2018

@author: ohaddoron
"""
from keras.preprocessing.image import ImageDataGenerator,load_img,img_to_array
from skimage import color as col
import os
import shutil
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation,Dropout,Conv2D,Flatten,MaxPooling2D
from sklearn.model_selection import train_test_split
from progressbar import ProgressBar
from keras.models import load_model
from skimage import io

path = './'
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
    pbar = ProgressBar()
    for label in pbar(os.listdir(path)):
        files = [file for file in os.listdir(path + '/' + label) if file.endswith('.jpeg')]
        for file in files:
            tmp = []
            img = io.imread(path + '/' + label + '/' + file)
            color = ('r','g','b')
            for i,c in enumerate(color):
                histr = np.histogram(img[:,:,i],bins=256)[0]
                tmp = np.hstack((tmp,histr))
                #img = cv2.resize(img,(200,200))
            #tmp.append(np.transpose(np.vstack(img.ravel())))
            X.append(np.transpose(np.vstack(tmp)))            
            y.append(int(label)-1)
    y -= np.min(y)
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
                if i > 30:
                    break  # otherwise the generator would loop indefinitely 

def trainMLP():
    pbar = ProgressBar()
    scores = []
    path = './busesRectSorted2'
    X,y = load(path)
    for i in pbar(range(1)):
        X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
        
        model = Sequential([
            Dense(500, input_shape=(X_train.shape[1],)),
            Activation('tanh'),
            # Dropout(0.5),
            # Dense(50,activation='tanh'),
            Dense(len(np.unique(y_train))),
            Activation('softmax'),
        ])
            
        model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
        
        model.fit(X_train, y_train, epochs=15, batch_size=100,shuffle=True,verbose=1)
        out = model.predict_classes(X_test)
        score = 0.0
        for o,yi in zip(out,y_test):
            if o == yi:
                score+=1
        scores.append(score/len(out)*100)
        print('\n' + str(scores[-1]))
    model.save('./models/RGB2.h5')
    return model
        

def convert_img(img):
    img = (255*img).astype(int)
    hsv_img = col.rgb2hsv(img)
    hsv_img += np.min(hsv_img)
    hsv_img /= np.max(hsv_img) * 255
    return hsv_img
    #return img
    



def trainCNN():
    train_datagen = ImageDataGenerator(
        # preprocessing_function=convert_img,
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
        )

    train_generator = train_datagen.flow_from_directory(
        './busesRectSorted2',
#	'./busesRect - Sorted',
        target_size=(150, 150),
        batch_size=15,
        class_mode='categorical')

    
    model = Sequential()
    # input: 100x100 images with 3 channels -> (100, 100, 3) tensors.
    # this applies 32 convolution filters of size 3x3 each.
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    # model.add(Dense(6, activation='softmax'))
    model.add(Dense(2, activation='softmax'))
    
    model.compile(loss='categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
    
    model.fit_generator(train_generator, epochs=10)
    
    # os.mkdir('./models')
    model.save('./models/RGB2.h5')
    #score = model.evaluate(x_test, y_test, batch_size=32)

def evalCNNModel(model):
    test_datagen = ImageDataGenerator(
        # preprocessing_function=convert_img,
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)


    test_generator = test_datagen.flow_from_directory(
        './busesRectOriginal2',
        #	'./busesRect - original',
        target_size=(150, 150),
        batch_size=1,
        class_mode='categorical')
    
    score = 0.0
    err_label = []
    count =0.0
    n_samples=600
    for i in range(n_samples):
        X,y = test_generator.next()
        if np.argmax(y) in [2,3]:
            count += 1
        if model.predict_classes(X)[0] == np.argmax(y):
            score+=1
        else:
            err_label.append(np.argmax(y))
    
    print '\nScore: ' + str(score/n_samples)

'''
model = trainCNN()
evalCNNModel(model)
'''
model = trainMLP()