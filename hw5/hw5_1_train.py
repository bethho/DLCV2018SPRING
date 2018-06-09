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
        
def load_data(video_path, data_path, base_model):
    od = getVideoList(data_path)
    video_category = od['Video_category']
    video_name = od['Video_name']
    labels = od['Action_labels']
    idx = np.arange(len(video_name))
    #np.random.shuffle(idx)
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
        label.append(int(labels[idx[i]]))
    print("time_cut_frame=%f, time_predict=%f" %(t12/len(video_name),t23/len(video_name)))
    return np.array(data), to_categorical(np.array(label), num_classes=11)

def classification_model():
    inputs = Input(shape=(2048,), name='DNN_input')
    #x = Dense(512, activation='relu', name='dense_1')(inputs)
    x = Dropout(0.5)(inputs)
    x = Dense(11, activation='softmax', name='DNN_output')(x)
    
    model = Model(inputs, x)
    model.compile(optimizer = Adam(lr = 1e-4), loss = 'categorical_crossentropy', metrics = ['accuracy'])
    return model

## model
input_tensor = Input(shape=(240, 320, 3))
#base_model = InceptionV3(input_tensor=input_tensor,weights='imagenet', include_top=False, pooling='avg')
base_model = ResNet50(input_tensor=input_tensor,weights='imagenet', include_top=False, pooling='avg')

## load DNN model
save = './model'
modelsave = os.path.join(save, 'model_{epoch:02d}-{val_acc:.2f}.h5')
if not os.path.exists(save):
    os.mkdir(save)
model = classification_model()
plot_model(model, to_file=os.path.join(save,'cassify_model.png'), show_shapes=True)
model.summary()
checkpoint = ModelCheckpoint(modelsave, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=True, mode='max')
# check 5 epochs
#early_stop = EarlyStopping(monitor='val_acc', patience=10, mode='max')
callbacks_list = [checkpoint]

train_data_path = './TrimmedVideos/label/gt_train.csv'
train_video_path = './TrimmedVideos/video/train'
val_data_path = './TrimmedVideos/label/gt_valid.csv'
val_video_path = './TrimmedVideos/video/valid'
train_data, train_labels = load_data(train_video_path, train_data_path, base_model)
val_data, val_labels = load_data(val_video_path, val_data_path, base_model)

train_history=model.fit(x=train_data,
                           y=train_labels,
                           validation_data = (val_data,val_labels),
                           epochs=1000, batch_size=128,
                           verbose=2, callbacks=callbacks_list)

save_csv(train_history, save)
show_train_history(train_history, 'acc', 'val_acc', save)
show_train_history(train_history, 'loss', 'val_loss', save)
