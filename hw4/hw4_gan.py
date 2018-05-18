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

## draw history
def plot_history(save):
    df = pd.read_csv('gan_train_history.csv')
    #df1 = pd.read_csv(os.path.join(save, 'kld.csv'))
    plt.figure(figsize=(20, 10))
    plt.subplot(121)
    plt.plot(df['D_loss'])
    plt.plot(df['G_loss'])
    plt.title('Train History')
    plt.ylabel('loss')
    plt.xlabel('Training step')
    plt.legend(['Discriminator','Generator'], loc='upper left')
    plt.subplot(122)
    plt.plot(df['D_acc'])
    #plt.plot(df['val_kld'])
    plt.title('Train History')
    plt.ylabel('acc')
    plt.xlabel('Training step')
    plt.legend(['Discrimator'], loc='upper left')
    plt.savefig(os.path.join(save, 'fig2_2.jpg'))
    plt.close()
    
## develop GAN model
class GAN():
    def __init__(self):
        self.img_rows = 64
        self.img_cols = 64
        self.channels = 3
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 512

        optimizer = Adam(0.0002, 0.5)

        # Build and compile the discriminator 
        # Build the generator
        self.discriminator, self.generator = self.build_model()
        self.discriminator.compile(loss='binary_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])

        # The generator takes noise as input and generates imgs
        z = Input(shape=(self.latent_dim,))
        img = self.generator(z)

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated images as input and determines validity
        valid = self.discriminator(img)

        # The combined model  (stacked generator and discriminator) takes
        # noise as input => generates images => determines validity
        self.combined = Model(z, valid)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)
        
    def build_model(self):
        image_size=64
        input_shape = (image_size, image_size, 3)
        batch_size = 128
        kernel_size = 3
        filters = 32
        latent_dim = 512

        # gan model = generator + discriminator
        # build discriminator model
        inputs = Input(shape=input_shape, name='discriminator_input')
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
        validity = Dense(1, activation='sigmoid', name='discriminator_output')(x)
        
        discriminator = Model(inputs, validity, name='discriminator')
        discriminator.summary()
        plot_model(discriminator, to_file='gan_cnn_discriminator.png', show_shapes=True)

        # build generator model
        latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
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
                                  name='generator_output')(x)
        print(outputs.shape)

        # instantiate generator model
        generator = Model(latent_inputs, outputs, name='generator')
        generator.summary()
        plot_model(generator, to_file='gan_cnn_generator.png', show_shapes=True)
        
        return discriminator, generator
    
    def train(self, epochs, batch_size=128, sample_interval=50, train_path='./hw4_data/train/'):

        # Load the dataset
        X_train = []
        a=os.listdir(train_path)
        a.sort(key = str.lower)
        for i in range(len(a)):
            fullpath = os.path.join(train_path, a[i])
            img = io.imread(fullpath)
            X_train.append(img.astype('float32') / 255)
        X_train = np.array(X_train)
        print(X_train.shape)
        
        half_batch = int(batch_size / 2)
        
        filename = './gan1/train_history.csv'
        if not os.path.exists(os.path.join(filename)):
            f = open(os.path.join(filename),'w')
            w = csv.writer(f)
            w.writerow(['epoch', 'D_loss','D_acc','G_loss'])        

        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random half batch of images
            idx = np.random.randint(0, X_train.shape[0], half_batch)
            imgs = X_train[idx]

            noise = np.random.normal(0, 1, (half_batch, self.latent_dim))

            # Generate a half batch of new images
            gen_imgs = self.generator.predict(noise)
            self.discriminator.trainable = True
            # Train the discriminator
            self.discriminator.summary()
            d_loss_real = self.discriminator.train_on_batch(imgs, np.ones((half_batch, 1)))
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, np.zeros((half_batch, 1)))
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)


            # ---------------------
            #  Train Generator
            # ---------------------

            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

            # The generator wants the discriminator to label the generated samples
            # as valid (ones)
            valid_y = np.array([1] * batch_size)
            self.discriminator.trainable = False
            self.combined.summary()
            # Train the generator
            g_loss = self.combined.train_on_batch(noise, valid_y)

            # Plot the progress
            w.writerows(np.array([epoch, d_loss[0], d_loss[1], g_loss]).reshape(1,-1))
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                save = './gan1'
                self.sample_images(epoch)
                model_name = 'model_'+str(epoch)+'.h5'
                self.generator.save(os.path.join(save, model_name))
        f.close()
        model_name = 'model_'+str(epochs)+'.h5'
        self.generator.save(os.path.join('./gan1', model_name))
        
    def sample_images(self, epoch):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, 512))
        gen_imgs = self.generator.predict(noise)

        # Rescale images 0 - 1
        gen_imgs = gen_imgs

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,:], cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("images1/face_%d.png" % epoch)
        plt.close()
        
## develope generator model
generator = load_model('./gan_model_50000.h5')

directory = sys.argv[1]
save = sys.argv[2]

## fig2_2.jpg
plot_history(save)

## fig2_3.jpg
np.random.seed(0)
r, c = 4, 8
noise = np.random.normal(0, 1, (r * c, 512))
gen_imgs = generator.predict(noise)
digit_size = 64
figure = np.zeros((digit_size * 4, digit_size * 8,3))
mu, sigma = 0, 1
z_sample = []
for i in range(4):
    for j in range(8):
        img_tt = gen_imgs[i*8+j]
        figure[i * digit_size: (i + 1) * digit_size,
        j * digit_size: (j + 1) * digit_size,:] = img_tt
io.imsave(os.path.join(save,'fig2_3.jpg'), figure)

