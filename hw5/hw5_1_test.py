import sys, argparse, os
from skimage import io
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from math import log, floor
from keras.utils import np_utils
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Activation, Flatten, Dropout, LeakyReLU, Input, UpSampling2D, merge, MaxPooling2D, Cropping2D
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Add, Embedding
from keras.layers import BatchNormalization, Concatenate, Lambda, Reshape, Conv2DTranspose, LeakyReLU
from keras import losses
from keras.losses import mse, binary_crossentropy
from keras import optimizers 
from keras.optimizers import SGD, Adam, Adadelta
from keras.utils import plot_model
import csv
from reader import getVideoList, readShortVideo
from keras.applications.inception_v3 import InceptionV3
from keras.applications.resnet50 import ResNet50
from keras.callbacks import ModelCheckpoint, EarlyStopping
import time

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
    t12 = 0
    t23 = 0
    for i in range(len(video_name)):
        t1 = time.time()
        frames = readShortVideo(video_path, video_category[idx[i]], video_name[idx[i]], downsample_factor=12, rescale_factor=1)
        t2 = time.time()
        features = base_model.predict(frames)
        t3 = time.time()
        t12 = t12+t2-t1
        t23 = t23+t3-t2
        data.append(np.average(features, axis=0))
    print("time_cut_frame=%f, time_predict=%f" %(t12/len(video_name),t23/len(video_name)))
    return np.array(data)
def classification_model():
    inputs = Input(shape=(2048,), name='DNN_input')
    #x = Dense(512, activation='relu', name='dense_1')(inputs)
    x = Dropout(0.5)(inputs)
    x = Dense(11, activation='softmax', name='DNN_output')(x)
    
    model = Model(inputs, x)
    model.compile(optimizer = Adam(lr = 1e-4), loss = 'categorical_crossentropy', metrics = ['accuracy'])
    return model

## load base model
input_tensor = Input(shape=(240, 320, 3))
base_model = ResNet50(input_tensor=input_tensor,weights='imagenet', include_top=False, pooling='avg')
base_model.summary()

## load DNN model
weights_path = './model_p1.h5'
model = classification_model()
model.load_weights(weights_path, by_name=True)

val_data_path = sys.argv[2] #'./TrimmedVideos/label/gt_valid.csv'
val_video_path = sys.argv[1] #'./TrimmedVideos/video/valid'
val_data = load_val_data(val_video_path, val_data_path, base_model)

## predict
pre = model.predict(val_data)
pre = np.argmax(pre, axis=1)
save_path = sys.argv[3]
F = open(os.path.join(save_path,'p1_valid.txt'),"w")
for i in range(pre.shape[0]):
    F.write(str(pre[i]))
    F.write('\n')
F.close()
