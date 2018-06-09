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

def classification_model():
    inputs = Input(shape=(2048,))
    x = Dense(512, activation='relu')(inputs)
    x = Dropout(0.5)(inputs)
    x = Dense(11, activation='softmax')(x)
    
    model = Model(inputs, x)
    model.compile(optimizer = Adam(lr = 1e-4), loss = 'categorical_crossentropy', metrics = ['accuracy'])
    return model

def RNN_model(cell='LSTM', sequence_size=10, return_sequence=True,dropout_rate=0.0):
    inputs = Input(shape=(None,2048), name='RNN_input')
    if cell == 'LSTM':
        RNN_cell = LSTM(512, return_sequences=return_sequence, dropout=dropout_rate, name='RNN_cell_lstm')
    elif cell == 'GRU':
        RNN_cell = GRU(512, return_sequences=return_sequence, dropout=dropout_rate, name='RNN_cell_gru')
    x = Bidirectional(RNN_cell, name='bidirection')(inputs)
    x = Dense(256, activation='relu', name='dense_1')(x)
    x = Dense(32, activation='relu', name='dense_2')(x)
    x = Dropout(dropout_rate)(x)
    x = Dense(11, activation='softmax', name='RNN_output')(x)    
    model = Model(inputs, x)
    model.compile(optimizer = Adam(lr = 1e-4), loss = 'categorical_crossentropy', metrics = ['accuracy'])
    return model
    
def model_predict(val_video_path, base_model, model, save_path):
    video_list = os.listdir(val_video_path)
    for i in video_list:
        F = open(os.path.join(save_path,i+'.txt'),"w")
        video_direct = os.path.join(val_video_path,i)
        features = load_img(video_direct, base_model, sequence_length=0, cut=False)
        features = np.expand_dims(np.array(features), axis=0)
        print(features.shape)
        predict = model.predict(features)
        predict = np.argmax(predict, axis=2)
        print(predict.shape)
        for i in range(predict.shape[1]):
            F.write(str(predict[0,i]))
            F.write('\n')
        F.close()
    return
    
def load_label(label_path, sequence_length, cut):
    text_file = open(label_path, "r")
    label = []
    labels = []
    list_txt = text_file.readlines()
    num = len(list_txt)
    for i in range(num):
        label.append(int(list_txt[i].strip('\n')))
    label = np.array(label)
    if cut == True:
        b = 0
        i = sequence_length
        shift = 50
        while i <= num:
            labels.append(label[i-sequence_length:i])
            i = i+shift
            b = b+1
        print("b=%d" %b)
        Y = to_categorical(np.array(labels), num_classes=11)
    else:
        Y = to_categorical(np.array(label), num_classes=11)
    return Y
    
def load_img(video_direct, base_model, sequence_length, cut):
    image_list = os.listdir(video_direct)
    image_list.sort(key = str.lower)
    features = []
    features_set = []
    count = 0
    num = len(image_list)
    for i in range(num):
        fullpath = os.path.join(video_direct, image_list[i])
        img = io.imread(fullpath)
        img = np.expand_dims(img, axis=0)
        feature = base_model.predict(img)
        features.append(feature[0])
    features = np.array(features)

    if cut == True:
        a = 0
        i = sequence_length
        shift = 50
        while i <= num:
            features_set.append(features[i-sequence_length:i,:])
            i = i+shift
            a = a+1
        print("a=%d" %a)
    else:
        features_set = features
    return np.array(features_set)
    
## load base model
input_tensor = Input(shape=(240, 320, 3))
base_model = ResNet50(input_tensor=input_tensor,weights='imagenet', include_top=False, pooling='avg')

## load RNN model
weights_path = './model_p3.h5'
model1 = RNN_model(cell='LSTM', return_sequence=True, dropout_rate=0.2)
model1.load_weights(weights_path, by_name=True)
val_video_path = sys.argv[1]
save_path = sys.argv[2]
model_predict(val_video_path, base_model, model1, save_path)
