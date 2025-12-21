import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import cv2
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
import matplotlib.image as mpimg
from imgaug import augmenters as iaa
import random

def getName(filePath):
    return filePath.split('\\')[-1]

def importDataInfo(path):
    columns = ['Center', 'Left', 'Right', 'Steering', 'Throttle', 'Brake', 'Speed']
    data = pd.read_csv(os.path.join(path, 'driving_log.csv'), names=columns)
    # Handle absolute paths if necessary, assuming relative for now
    data['Center'] = data['Center'].apply(getName)
    print('Total Images Imported', data.shape[0])
    return data

def balanceData(data, display=True):
    nBin = 31
    samplesPerBin = 500  # Confirmed optimal for your 25k images
    hist, bins = np.histogram(data['Steering'], nBin)
    
    if display:
        center = (bins[:-1] + bins[1:]) * 0.5
        plt.bar(center, hist, width=0.06)
        plt.plot((np.min(data['Steering']), np.max(data['Steering'])), (samplesPerBin, samplesPerBin))
        plt.show()
    
    removeindexList = []
    for j in range(nBin):
        binDataList = []
        for i in range(len(data['Steering'])):
            if data['Steering'][i] >= bins[j] and data['Steering'][i] <= bins[j + 1]:
                binDataList.append(i)
        binDataList = shuffle(binDataList)
        binDataList = binDataList[samplesPerBin:]
        removeindexList.extend(binDataList)
    
    data.drop(data.index[removeindexList], inplace=True)
    print('Remaining Images:', len(data))
    return data

def loadData(path, data):
    imagesPath = []
    steering = []
    # AGGRESSIVE CORRECTION to ensure turns are learned
    correction = 0.5 

    for i in range(len(data)):
        indexed_data = data.iloc[i]
        
        # Center
        imagesPath.append(os.path.join(path, 'IMG', indexed_data[0]))
        steering.append(float(indexed_data[3]))
        
        # Left (Steer Right)
        imagesPath.append(os.path.join(path, 'IMG', indexed_data[1].split('\\')[-1]))
        steering.append(float(indexed_data[3]) + correction)
        
        # Right (Steer Left)
        imagesPath.append(os.path.join(path, 'IMG', indexed_data[2].split('\\')[-1]))
        steering.append(float(indexed_data[3]) - correction)
        
    imagesPath = np.asarray(imagesPath)
    steering = np.asarray(steering)
    return imagesPath, steering

def augmentImage(imgPath, steering):
    img = mpimg.imread(imgPath)
    
    # PAN/SHIFT
    if np.random.rand() < 0.5:
        pan = iaa.Affine(translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)})
        img = pan.augment_image(img)
    
    # ZOOM (Critical for variation)
    if np.random.rand() < 0.5:
        zoom = iaa.Affine(scale=(1, 1.2))
        img = zoom.augment_image(img)
        
    # BRIGHTNESS
    if np.random.rand() < 0.5:
        brightness = iaa.Multiply((0.5, 1.2))
        img = brightness.augment_image(img)
        
    # FLIP
    if np.random.rand() < 0.5:
        img = cv2.flip(img, 1)
        steering = -steering
        
    return img, steering

def preProcessing(img):
    # CROP: Remove hood (bottom 25px) and sky (top 60px)
    img = img[60:135,:,:] 
    # COLOR: NVIDIA Standard
    img = cv2.cvtColor(img, cv2.COLOR_RGB2YUV) 
    # BLUR: Reduce noise
    img = cv2.GaussianBlur(img, (3, 3), 0)
    # RESIZE: Input shape model expects
    img = cv2.resize(img, (200, 66))
    # NORMALIZE
    img = img / 255
    return img

def batchGen(imagesPath, steeringList, batchSize, trainFlag):
    while True:
        imgBatch = []
        steeringBatch = []
        
        for i in range(batchSize):
            index = random.randint(0, len(imagesPath) - 1)
            if trainFlag:
                img, steering = augmentImage(imagesPath[index], steeringList[index])
            else:
                img = mpimg.imread(imagesPath[index])
                steering = steeringList[index]
            
            img = preProcessing(img)
            imgBatch.append(img)
            steeringBatch.append(steering)
        yield (np.asarray(imgBatch), np.asarray(steeringBatch))

def createModel():
    model = Sequential()
    # Explicit Input Layer
    model.add(Input(shape=(66, 200, 3)))
    
    model.add(Convolution2D(24, (5, 5), strides=(2, 2), activation='elu'))
    model.add(Convolution2D(36, (5, 5), strides=(2, 2), activation='elu'))
    model.add(Convolution2D(48, (5, 5), strides=(2, 2), activation='elu'))
    model.add(Convolution2D(64, (3, 3), activation='elu'))
    model.add(Convolution2D(64, (3, 3), activation='elu'))
    
    model.add(Flatten())
    model.add(Dense(100, activation='elu'))
    model.add(Dropout(0.5)) # Prevent overfitting
    model.add(Dense(50, activation='elu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='elu'))
    model.add(Dense(1))
    
    model.compile(Adam(learning_rate=1e-4), loss='mse')
    return model