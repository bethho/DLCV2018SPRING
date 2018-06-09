import sys, argparse, os
from skimage import io
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from math import log, floor
from keras.utils import np_utils
import csv
import time
from reader import getVideoList, readShortVideo
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Activation, Flatten, Dropout, LeakyReLU, Input, UpSampling2D, merge, MaxPooling2D, Cropping2D
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Add, Embedding
from keras.layers import BatchNormalization, Concatenate, Lambda, Reshape, Conv2DTranspose, LeakyReLU
from keras.layers import Bidirectional, GRU, LSTM
from keras import losses
from keras.losses import mse, binary_crossentropy
from keras import optimizers 
from keras.optimizers import SGD, Adam, Adadelta
from keras.utils import plot_model
from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet50 import ResNet50
from keras.callbacks import ModelCheckpoint, EarlyStopping

def show_train_history(train_history,train,validation, save):
    fig = plt.gcf()
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[validation])
    plt.title('Train History')
    plt.ylabel(train)
    plt.xlabel('Epoch')
    plt.legend(['train','validation'], loc='upper left')
    plt.draw()
    fig.savefig(os.path.join(save, train+'.png'))
    plt.close()

def save_csv(train_history1, save1):
    A = train_history1.history['acc']
    B = train_history1.history['val_acc']
    AB = np.array([A,B]).T
    #S = np.max(AB,0)
    C = train_history1.history['loss']
    D = train_history1.history['val_loss']
    CD = np.array([C, D]).T

    if not os.path.exists(os.path.join(save1, 'acc.csv')):
        f = open(os.path.join(save1, 'acc.csv'),'w')
        w = csv.writer(f)
        w.writerow(['acc', 'val_acc'])
        w.writerows(AB)
        f.close()
    if not os.path.exists(os.path.join(save1, 'loss.csv')):
        f = open(os.path.join(save1, 'loss.csv'),'w')
        w = csv.writer(f)
        w.writerow(['loss', 'val_loss'])
        w.writerows(CD)
        f.close()
        
def load_val_data(video_path, data_path, base_model):
    od = getVideoList(data_path)
    video_category = od['Video_category']
    video_name = od['Video_name']
    idx = np.arange(len(video_name))
    data = []
    label = []
    for i in range(len(video_name)):
        frames = readShortVideo(video_path, video_category[idx[i]], video_name[idx[i]], downsample_factor=12, rescale_factor=1)
        features = base_model.predict(frames)
        data.append(features)
    return np.array(data)

def classification_model():
    inputs = Input(shape=(2048,))
    x = Dense(512, activation='relu')(inputs)
    x = Dropout(0.5)(inputs)
    x = Dense(11, activation='softmax')(x)
    
    model = Model(inputs, x)
    model.compile(optimizer = Adam(lr = 1e-4), loss = 'categorical_crossentropy', metrics = ['accuracy'])
    return model

def RNN_model(cell='LSTM', return_sequence=False, dropout_rate=0.0):
    inputs = Input(shape=(None,2048))
    if cell == 'LSTM':
        RNN_cell = LSTM(512, return_sequences=return_sequence, dropout=dropout_rate)
    elif cell == 'GRU':
        RNN_cell = GRU(512, return_sequences=return_sequence, dropout=dropout_rate)
    x = Bidirectional(RNN_cell, name='bidirection')(inputs)
    x = Dense(256, activation='relu', name='dense_1')(x)
    #x = Dense(32, activation='relu', name='dense_2')(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(11, activation='softmax', name='RNN_output')(x)    
    model = Model(inputs, x)
    model.compile(optimizer = Adam(lr = 1e-4), loss = 'categorical_crossentropy', metrics = ['accuracy'])
    return model
    
## load base model
input_tensor = Input(shape=(240, 320, 3))
base_model = ResNet50(input_tensor=input_tensor,weights='imagenet', include_top=False, pooling='avg')

## load RNN model
weights_path = './model_p2.h5'
model1 = RNN_model(cell='LSTM', return_sequence=False, dropout_rate=0.2)
model1.load_weights(weights_path, by_name=True)

##load valid/test data
val_video_path = sys.argv[1]
val_data_path = sys.argv[2]
val_data_1 = load_val_data(val_video_path, val_data_path, base_model)
Max = 234
X_val_1 = pad_sequences(val_data_1, dtype='float32', maxlen=Max, padding='post')
pre = model1.predict(X_val_1)
pre = np.argmax(pre, axis=1)

## record predict result
save_path = sys.argv[3]
F = open(os.path.join(save_path,'p2_valid.txt'),"w")
for i in range(pre.shape[0]):
    F.write(str(pre[i]))
    F.write('\n')
F.close()

