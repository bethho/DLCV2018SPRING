import sys, argparse, os
from skimage import io
import numpy as np
import tensorflow as tf
#os.environ["CUDA_VISIBLE_DEVICES"]="0"
#from keras import backend as K
#config = tf.ConfigProto()
#config.gpu_options.allow_growth=True
#sees = tf.Session(config=config)
#K.set_session(sees)
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Activation, Flatten, Dropout, LeakyReLU, Input, UpSampling2D, merge, MaxPooling2D, Cropping2D
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D
from keras.layers import BatchNormalization
from keras import losses
from keras import optimizers 
from keras.optimizers import SGD, Adam, Adadelta

def output_mask(predict,filename):
    color = np.array([[0,1,1],
                      [1,1,0],
                      [1,0,1],
                      [0,1,0],
                      [0,0,1],
                      [1,1,1],
                      [0,0,0]])
    out_mask = (color[predict]*255).astype(np.uint8)
    io.imsave(filename, out_mask)
    
def predict(model, path, filename, save):
    fullpath = os.path.join(path, filename)
    img = io.imread(fullpath)
    img = (img.astype('float32') / 255)
    img = np.expand_dims(img, axis=0)
    predict = model.predict(img)
    predict = np.argmax(predict, axis=3)
    output_mask(predict[0,:,:], os.path.join(save, filename.split('_')[0]+'_mask.png'))

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

VGG16 = model_VGG16()
#VGG16.summary()
layer_name = 'block5_pool'
#x = VGG16.get_layer(layer_name).output
x = VGG16.output
output = Conv2D(7, 2, activation = 'relu', padding = 'same', name='block6_conv')(UpSampling2D(size = (32,32), name='block6_up_sampling')(x))
FCNN32 = Model(inputs=VGG16.input, outputs=output)
weights_path = './FCNN32_model_best.h5'
FCNN32.load_weights(weights_path, by_name=True)
#FCNN32.summary()

path = sys.argv[1]#'./data/validation/'
save = sys.argv[2]#'./data/predict_result/FCNN32'
if not os.path.exists(save):
    os.mkdir(save)
a=os.listdir(path)
for i in range(len(a)):
    if (a[i].endswith("sat.jpg")):
        predict(FCNN32, path, a[i], save)
