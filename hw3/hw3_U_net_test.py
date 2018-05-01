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

def U_net_VGG16(w_size, h_size, weights_path):
    VGG16 = model_VGG16()
    x = VGG16.get_layer('block5_conv3').output
    print(x.shape)
    ## Block 6 up_sample
    up6 = Conv2D(512, 2, activation = 'relu', padding = 'same', name='block6_conv1')(UpSampling2D(size = (2,2), name='block6_up')(x))
    print(up6.shape)
    conv4 = VGG16.get_layer('block4_conv3').output
    print(conv4.shape)
    merge6 = Concatenate(name='block6_concat')([conv4,up6])
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', name='block6_conv2')(merge6)
    conv6 = Conv2D(512, 3, activation = 'relu', padding = 'same', name='block6_conv3')(conv6)
    
    ## Block 7 up_sampling
    up7 = Conv2D(256, 2, activation = 'relu', padding = 'same', name='block7_conv1')(UpSampling2D(size = (2,2), name='block7_up')(conv6))
    conv3 = VGG16.get_layer('block3_conv3').output
    merge7 = Concatenate(name='block7_concat')([conv3,up7])
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', name='block7_conv2')(merge7)
    conv7 = Conv2D(256, 3, activation = 'relu', padding = 'same', name='block7_conv3')(conv7)

    ## Block 8 up_sampling
    up8 = Conv2D(128, 2, activation = 'relu', padding = 'same', name='block8_conv1')(UpSampling2D(size = (2,2), name='block8_up')(conv7))
    conv2 = VGG16.get_layer('block2_conv2').output
    merge8 = Concatenate(name='block8_concat')([conv2,up8])
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', name='block8_conv2')(merge8)
    conv8 = Conv2D(128, 3, activation = 'relu', padding = 'same', name='block8_conv3')(conv8)

    ## Block 9 up_sampling
    up9 = Conv2D(64, 2, activation = 'relu', padding = 'same', name='block9_conv1')(UpSampling2D(size = (2,2), name='block9_up')(conv8))
    conv1 = VGG16.get_layer('block1_conv2').output
    merge9 = Concatenate(name='block9_concat')([conv1,up9])
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', name='block9_conv2')(merge9)
    conv9 = Conv2D(64, 3, activation = 'relu', padding = 'same', name='block9_conv3')(conv9)
    conv9 = Conv2D(16, 3, activation = 'relu', padding = 'same', name='block10_conv1')(conv9)
    conv10 = Conv2D(7, 1, activation = 'softmax', name='predictions')(conv9)

    U_net = Model(inputs=VGG16.input, outputs=conv10)
    
    if weights_path is not None:
        U_net.load_weights(weights_path, by_name=True)
    U_net.compile(optimizer = Adam(lr = 1e-4), loss = 'categorical_crossentropy', metrics = ['accuracy'])
    return U_net

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

weights_path = './U_net_VGG16_model_best.h5'
U_net = U_net_VGG16(512, 512, weights_path)
U_net.summary()

path = sys.argv[1]
save = sys.argv[2]
if not os.path.exists(save):
    os.mkdir(save)
a=os.listdir(path)

for i in range(len(a)):
    if (a[i].endswith("sat.jpg")):
        print(a[i])
        predict(U_net, path, a[i], save)
