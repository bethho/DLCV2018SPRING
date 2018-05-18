import sys, argparse, os
from skimage import io
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from math import log, floor
from keras.utils import np_utils
from keras import backend as K
#os.environ["CUDA_VISIBLE_DEVICES"]="0"
#config = tf.ConfigProto()
#config.gpu_options.allow_growth=True
#sees = tf.Session(config=config)
#K.set_session(sees)
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model, load_model
from keras.layers import Dense, Activation, Flatten, Dropout, LeakyReLU, Input, UpSampling2D, merge, MaxPooling2D, Cropping2D
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, ZeroPadding2D, Add
from keras.layers import BatchNormalization, Concatenate, Lambda, Reshape, Conv2DTranspose
from keras import losses
from keras.losses import mse, binary_crossentropy
from keras import optimizers 
from keras.optimizers import SGD, Adam, Adadelta
from keras.utils import plot_model
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE

## draw history
def plot_history(save):
    df = pd.read_csv('mean_squared_error.csv')
    df1 = pd.read_csv('kld.csv')
    plt.figure(figsize=(20, 10))
    plt.subplot(121)
    plt.plot(df['mean_squared_error'])
    plt.plot(df['val_mean_squared_error'])
    plt.title('Train History')
    plt.ylabel('mean_squared_error')
    plt.xlabel('Epoch')
    plt.legend(['train','validation'], loc='upper left')
    plt.subplot(122)
    plt.plot(df1['kld'])
    plt.plot(df1['val_kld'])
    plt.title('Train History')
    plt.ylabel('KLD')
    plt.xlabel('Epoch')
    plt.legend(['train','validation'], loc='upper left')
    plt.savefig(os.path.join(save, 'fig1_2.jpg'))
    plt.close()

## vae_model
def sampling(args):
    """Reparameterization trick by sampling fr an isotropic unit Gaussian.
    # Arguments:
        args (tensor): mean and log of variance of Q(z|X)
    # Returns:
        z (tensor): sampled latent vector
    """

    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon
def vae(image_size, la):
    print(image_size)
    input_shape = (image_size, image_size, 3)
    batch_size = 128
    kernel_size = 3
    filters = 32
    latent_dim = 512
    epochs = 30
    
    # VAE model = encoder + decoder
    # build encoder model
    inputs = Input(shape=input_shape, name='encoder_input')
    x = inputs
    for i in range(3):
        filters *= 2
        x = Conv2D(filters=filters,
                   kernel_size=kernel_size,
                   activation='relu',
                   strides=2,
                   padding='same')(x)
    
    # shape info needed to build decoder model
    shape = K.int_shape(x)
    
    # generate latent vector Q(z|X)
    x = Flatten()(x)
    x = Dense(1024, activation='relu')(x)
    z_mean = Dense(latent_dim, name='z_mean')(x)
    z_log_var = Dense(latent_dim, name='z_log_var')(x)
    
    # use reparameterization trick to push the sampling out as input
    # note that "output_shape" isn't necessary with the TensorFlow backend
    z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])
    # instantiate encoder model
    encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
    encoder.summary()
    plot_model(encoder, to_file='vae_cnn_encoder.png', show_shapes=True)
    
    # build decoder model
    latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
    print(shape[1])
    x = Dense(shape[1] * shape[2] * shape[3], activation='relu')(latent_inputs)
    x = Reshape((shape[1], shape[2], shape[3]))(x)
    print(x.shape)
    for i in range(3):
        x = Conv2DTranspose(filters=filters,
                            kernel_size=kernel_size,
                            activation='relu',
                            strides=2,
                            padding='same')(x)
        print(x.shape)
        filters //= 2
    
    outputs = Conv2DTranspose(filters=3,
                              kernel_size=kernel_size,
                              activation='sigmoid',
                              padding='same',
                              name='decoder_output')(x)
    print(outputs.shape)
    
    # instantiate decoder model
    decoder = Model(latent_inputs, outputs, name='decoder')
    decoder.summary()
    plot_model(decoder, to_file='vae_cnn_decoder.png', show_shapes=True)
    
    # instantiate VAE model
    outputs = decoder(encoder(inputs)[2])
    print(outputs.shape)
    vae = Model(inputs, outputs, name='vae')
    plot_model(vae, to_file='vae_cnn.png', show_shapes=True)
    vae.summary()
    def vae_loss(inputs, outputs):
        mse = K.mean(K.square(outputs - inputs))
        kl_loss = -0.5*K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        kl_loss = la*kl_loss
        vae_loss = K.mean(mse + kl_loss)
        return vae_loss
        
    def kld(z_log_var, z_mean):
        kld = -0.5*la*K.sum(1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1)
        return kld
    vae.compile(optimizer='rmsprop', loss=vae_loss, metrics=['mse', kld])    
    return vae

## develope vae_model
image_size = 64
print(image_size)
weights_path ='vae_model.h5'
vae = vae(image_size, la=1e-4)
vae.load_weights(weights_path, by_name=True)

directory = sys.argv[1]
save = sys.argv[2]
## test_img
path = os.path.join(directory,'test')
img_test = []
a=os.listdir(path)
a.sort(key = str.lower)
for i in range(len(a)):
    fullpath = os.path.join(path, a[i])
    img = io.imread(fullpath)
    img_test.append(img.astype('float32') / 255)
img_test = np.array(img_test)

## test_label
label_path=os.path.join(directory,'test.csv')
df = pd.read_csv(label_path)
gender = df['Male']

## fig1_2.jpg
plot_history(save)

np.random.seed(0)
## fig1_3.jpg
digit_size = 64
figure = np.zeros((digit_size * 2, digit_size * 10,3))
a = np.arange(img_test.shape[0])
np.random.shuffle(a)
for j in range(10):
    img_tt = vae.predict(img_test[a[j]:a[j]+1])
    figure[0 * digit_size: (0 + 1) * digit_size,
    j * digit_size: (j + 1) * digit_size,:] = img_test[a[j]]
    figure[1 * digit_size: (1 + 1) * digit_size,
    j * digit_size: (j + 1) * digit_size,:] = img_tt
io.imsave(os.path.join(save,'fig1_3.jpg'), figure)

## fig1_4.jpg
# figure with 64x64 digits
decoder = Model(vae.layers[2].inputs, vae.layers[2].outputs)
digit_size = 64
r, c = 4, 8
figure = np.zeros((digit_size * r, digit_size * c,3))
z_sample = np.random.normal(0, 1, (r * c, 512))
for i in range(4):
    for j in range(8):
        img_tt = decoder.predict(z_sample[i*8+j].reshape(1,-1))
        figure[i * digit_size: (i + 1) * digit_size,
        j * digit_size: (j + 1) * digit_size,:] = img_tt
io.imsave(os.path.join(save,'fig1_4.jpg'), figure)

## fig1_5.jpg
num = 1000
ind = np.random.randint(0, img_test.shape[0], (num,))
img_vis = img_test[ind].reshape(num,64*64*3)
gender_vis = gender[ind]
y = np.array(gender_vis)
x = np.array(img_vis,dtype=np.float64)
# perform t-SNE embedding
vis_data = TSNE(n_components=2, random_state=0).fit_transform(x)
# plot the result
vis_x = vis_data[:,0]
vis_y = vis_data[:,1]
fig = plt.figure(figsize=(8, 5), dpi=80)
axes = plt.subplot(111)
type1 = axes.scatter(vis_x[y==0], vis_y[y==0], s=20, c='red')
type2 = axes.scatter(vis_x[y==1], vis_y[y==1], s=20, c='green')
axes.legend((type1, type2), ('Female','Male'), loc=2)
fig.savefig(os.path.join(save,"fig1_5.jpg"))
