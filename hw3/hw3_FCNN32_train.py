import sys, argparse, os
from skimage import io
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.utils import np_utils
#os.environ["CUDA_VISIBLE_DEVICES"]="1"
#from keras import backend as K
#config = tf.ConfigProto()
#config.gpu_options.allow_growth=True
#sees = tf.Session(config=config)
#K.set_session(sees)

def mask(A, ind):
    C = np.zeros((A.shape[0],A.shape[1]))
    C[A==ind] = 1
    return C

def labels(label):
    labels = np.zeros((label.shape[0],label.shape[1],label.shape[2],7))
    for i in range(label.shape[0]):
        A = label[i,:,:,0]+2*label[i,:,:,1]+2+4*label[i,:,:,2]+4
        labels[i,:,:,0] = mask(A, 12) ## urban (0,1,1)
        labels[i,:,:,1] = mask(A, 9)  ## agriculture (1,1,0)
        labels[i,:,:,2] = mask(A, 11) ## rangeland (1,0,1)
        labels[i,:,:,3] = mask(A, 8)  ## forest (0,1,0)
        labels[i,:,:,4] = mask(A, 10) ## water (0,0,1)
        labels[i,:,:,5] = mask(A, 13) ## barren (1,1,1)
        labels[i,:,:,6] = mask(A, 6)  ## unknown (0,0,0)   
    return labels

def load_data(path):
    data_list = []
    mask_list = []
    a=os.listdir(path)
    a.sort(key = str.lower)
    for i in range(len(a)):
        if (a[i].endswith("mask.png")):
            fullpath = os.path.join(path, a[i])
            mask_list.append(fullpath)
        if (a[i].endswith("sat.jpg")):
            fullpath = os.path.join(path, a[i])
            data_list.append(fullpath)
    return data_list, mask_list

def dataGenerator(data_list, label_list, batch_size):
    state_idx = 0
    while True:
        data = []
        mask = []
        if state_idx == 0:
            idx = np.arange(len(data_list))
            np.random.shuffle(idx)
            x = (np.asarray(data_list)[idx]).tolist()
            y = (np.asarray(label_list)[idx]).tolist()
        for num in range(batch_size):
            img = io.imread(x[state_idx])
            data.append(img.astype('float32') / 255.)
            img = io.imread(y[state_idx])
            mask.append(img.astype('float32') / 255.)
            state_idx += 1
            if state_idx >=  (len(data_list)//batch_size)*batch_size:
                state_idx = 0
        mask = labels(np.array(mask))
        yield np.array(data), np.array(mask)

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


from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Activation, Flatten, Dropout, LeakyReLU, Input, UpSampling2D, merge, MaxPooling2D, Cropping2D
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Add
from keras.layers import BatchNormalization, Concatenate
from keras import losses
from keras import optimizers 
from keras.optimizers import SGD, Adam, Adadelta

def model_VGG16():
    size = 512
    inputs = Input(shape=(size, size,3))
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1')(inputs)
    x = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)
    
    #flatten
    #x = Flatten(name='flatten')(x)
    #x = Dense(4096, activation='relu', name='fc1')(x)
    #x = Dense(4096, activation='relu', name='fc2')(x)
    #x = Dense(1000, activation='softmax', name='predictions')(x)
    
    # Create model
    model = Model(inputs, x)

    #load_weight
    weights_path = './vgg16_weights_tf_dim_ordering_tf_kernels.h5'
    model.load_weights(weights_path, by_name=True)
    
    return model

## develope FCNN32 model
VGG16 = model_VGG16()
#VGG16.summary()
layer_name = 'block5_pool'
#x = VGG16.get_layer(layer_name).output
x = VGG16.output
output = Conv2D(7, 2, activation = 'relu', padding = 'same', name='block6_conv')(UpSampling2D(size = (32,32), name='block6_up_sampling')(x))
FCNN32 = Model(inputs=VGG16.input, outputs=output)
FCNN32.compile(optimizer = Adam(lr = 1e-4), loss = 'categorical_crossentropy', metrics = ['accuracy'])
FCNN32.summary()

## load data
path = '../data/train'
data_list, mask_list = load_data(path)
batch_size=2
train_data = dataGenerator(data_list, mask_list, batch_size)

val_data = []
val_label = []
path = '../data/validation/'
a=os.listdir(path)
a.sort(key = str.lower)
for i in range(len(a)):
    if (a[i].endswith("mask.png")):
        fullpath = os.path.join(path, a[i])
        img = io.imread(fullpath)
        val_label.append(img.astype('float32') / 255)
    if (a[i].endswith("sat.jpg")):
        fullpath = os.path.join(path, a[i])
        img = io.imread(fullpath)
        val_data.append(img.astype('float32') / 255)

val_data = np.array(val_data)
val_label = np.array(val_label)
val_labels = labels(val_label)
print(val_data.shape)
print(val_labels.shape)


## train
from keras.callbacks import ModelCheckpoint, EarlyStopping
save = './model_FCNN32'
modelsave = os.path.join(save, 'model_{epoch:02d}-{val_acc:.2f}.h5')
if not os.path.exists(save):
    os.mkdir(save)    
checkpoint = ModelCheckpoint(modelsave, monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=True, mode='max')
# check 5 epochs
#early_stop = EarlyStopping(monitor='val_acc', patience=10, mode='max')
callbacks_list = [checkpoint]


train_history = FCNN32.fit_generator(train_data,
        epochs = 40,
        #validation_data = val_data,
        validation_data = (val_data, val_labels),
        steps_per_epoch = int(len(data_list) // batch_size),
        #validation_steps= int(len(data_list) // batch_size),
        callbacks = [checkpoint])

show_train_history(train_history, 'acc', 'val_acc', './')
show_train_history(train_history, 'loss', 'val_loss', './')
