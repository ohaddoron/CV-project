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
import imgaug as ia
from imgaug import augmenters as iaa
from skimage import io
from matplotlib import pyplot as plt
from matplotlib import patches as patches
from joblib import Parallel,delayed
import csv
plt.ioff()

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
    

def getSeq():
    # Sometimes(0.5, ...) applies the given augmenter in 50% of all cases,
    # e.g. Sometimes(0.5, GaussianBlur(0.3)) would blur roughly every second image.
    sometimes = lambda aug: iaa.Sometimes(0.5, aug)
    
    # Define our sequence of augmentation steps that will be applied to every image
    # All augmenters with per_channel=0.5 will sample one value _per image_
    # in 50% of all cases. In all other cases they will sample new values
    # _per channel_.
    seq = iaa.Sequential(
        [
            # apply the following augmenters to most images
            iaa.Fliplr(0.5), # horizontally flip 50% of all images
            iaa.Flipud(0.2), # vertically flip 20% of all images
            # crop images by -5% to 10% of their height/width
            # sometimes(iaa.CropAndPad(
            #     percent=(-0.05, 0.1),
            #     pad_mode=ia.ALL,
            #     pad_cval=(0, 255)
            # )),
            sometimes(iaa.Affine(
                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}, # scale images to 80-120% of their size, individually per axis
                translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)}, # translate by -20 to +20 percent (per axis)
                rotate=(-45, 45), # rotate by -45 to +45 degrees
                shear=(-16, 16), # shear by -16 to +16 degrees
                order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
                cval=(0, 255), # if mode is constant, use a cval between 0 and 255
                mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)
            )),
            # execute 0 to 5 of the following (less important) augmenters per image
            # don't execute all of them, as that would often be way too strong
            iaa.SomeOf((0, 5),
                [
                    # sometimes(iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200))), # convert images into their superpixel representation
                    iaa.OneOf([
                        iaa.GaussianBlur((0, 3.0)), # blur images with a sigma between 0 and 3.0
                        iaa.AverageBlur(k=(2, 7)), # blur image using local means with kernel sizes between 2 and 7
                        iaa.MedianBlur(k=(3, 11)), # blur image using local medians with kernel sizes between 2 and 7
                    ]),
                    iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)), # sharpen images
                    iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)), # emboss images
                    # search either for all edges or for directed edges,
                    # blend the result with the original image using a blobby mask
                    iaa.SimplexNoiseAlpha(iaa.OneOf([
                        iaa.EdgeDetect(alpha=(0.5, 1.0)),
                        iaa.DirectedEdgeDetect(alpha=(0.5, 1.0), direction=(0.0, 1.0)),
                    ])),
                    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5), # add gaussian noise to images
                    iaa.OneOf([
                        iaa.Dropout((0.01, 0.1), per_channel=0.5), # randomly remove up to 10% of the pixels
                        iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2),
                    ]),
                    # iaa.Invert(0.05, per_channel=True), # invert color channels
                    iaa.Add((-10, 10), per_channel=0.5), # change brightness of images (by -10 to 10 of original value)
                    # iaa.AddToHueAndSaturation((-20, 20)), # change hue and saturation
                    # either change the brightness of the whole image (sometimes
                    # per channel) or change the brightness of subareas
                    iaa.OneOf([
                        iaa.Multiply((0.5, 1.5), per_channel=0.5),
                        iaa.FrequencyNoiseAlpha(
                            exponent=(-4, 0),
                            first=iaa.Multiply((0.5, 1.5), per_channel=True),
                            second=iaa.ContrastNormalization((0.5, 2.0))
                        )
                    ]),
                    iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5), # improve or worsen the contrast
                    iaa.Grayscale(alpha=(0.0, 1.0)),
                    sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)), # move pixels locally around (with random strengths)
                    sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))), # sometimes move parts of the image around
                    sometimes(iaa.PerspectiveTransform(scale=(0.01, 0.1)))
                ],
                random_order=True
            )
        ],
        random_order=True
    )
    return seq



def iterate_database_and_generate():
    locs = {}
    with open('./busesTrain/annotationsTrain.txt', 'r') as annFileGT:
         for line in annFileGT:
            imName = line.split(':')[0]
            # bus = os.path.join(settings.busOriginalDir, imName)
            locs[imName] = []
            for loc in line.split(':')[1].split('],'):
                loc = [int(i) for i in loc.replace('[','').replace(']','').split(',')]
                locs[imName].append(loc)
    bb = Parallel(n_jobs=6)(delayed(augment)(key,locs[key]) for key in locs)
    '''
    bb = []
    for key in locs:
        bb.append(augment(key,locs[key]))
    '''
    np.save('AugmentatedAnnotations.npy',bb)

    '''bb = []
    for key in locs:
        bb.append(augment(key,locs[key]))'''
    return bb
        
    
def augment(key,locs):
    image = io.imread('./busesTrain/' + key)

    for k in range(150):
        bb = []
        seq = getSeq()
        seq_det = seq.to_deterministic()
        
        annotations_wh = np.vstack(locs)[:]
        annotations_xy = annotations_wh[:]
        annotations_xy[:,2] += annotations_xy[:,0]
        annotations_xy[:,3] += annotations_xy[:,1]
        bbs = ia.BoundingBoxesOnImage([ia.BoundingBox(*sub) for sub in annotations_xy],
                                       shape=image.shape)
        
        # fig,ax = plt.subplots(1)
        image_aug = seq_det.augment_images([image])[0]
        bbs_aug = seq_det.augment_bounding_boxes([bbs])[0]
        io.imsave('./Augmented_Buses/' + key.replace('.JPG','') + '_' + str(k)+'.jpg',image_aug)
        # plt.imshow(image_aug)
        for b in bbs_aug.bounding_boxes:
            x_min = int(b.x1)
            y_min = int(b.y1)
            x_max = int(b.x2)
            y_max = int(b.y2)
            label = hash_map(int(b.label))
            bb.append([key.replace('.JPG','_'+str(k)+'.jpg'),x_min,x_max,y_min,y_max,label])
            '''
            rect = patches.Rectangle((x,y),w,h,
                                     linewidth=1,
                                     edgecolor='r',
                                     facecolor='None',
                                     fill=False)'''
            # ax.add_patch(rect)
        # fig.savefig('./Augmented_Buses_marked/' + key + '_' + str(k) +'.jpg')
        # plt.close('all')
        # bb = (bb,)
        for row in (bb,):
            with open('AugmentatedAnnotations.csv','a+') as f:
                writer=csv.writer(f)
                writer.writerows(row)
            
    return bb

def hash_map(num):
    if num == 1:
        return 'green'
    if num == 2:
        return 'yellow
    if num == 3:
        return 'white'
    if num == 4:
        return 'grey'
    if num == 5:
        return 'blue'
    if num ==6 :
        return 'red'        
        
# settings,params = settings_params.load()
# genTrainingData(settings,params)

locs = {}
with open('./busesTrain/annotationsTrain.txt', 'r') as annFileGT:
    for line in annFileGT:
        imName = line.split(':')[0]
        # bus = os.path.join(settings.busOriginalDir, imName)
        locs[imName] = []
        for loc in line.split(':')[1].split('],'):
            loc = [int(i) for i in loc.replace('[', '').replace(']', '').split(',')]
            locs[imName].append(loc)
