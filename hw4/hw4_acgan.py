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
import pandas as pd

## draw history
def plot_history(save):
    df = pd.read_csv('acgan_train_history.csv')
    #df1 = pd.read_csv(os.path.join(save, 'kld.csv'))
    plt.figure(figsize=(20, 10))
    plt.subplot(121)
    plt.plot(df['D_atribute_loss_real'])
    plt.plot(df['D_atribute_loss_fake'])
    plt.title('loss of Atribute Classification')
    plt.ylabel('loss')
    plt.xlabel('Training step')
    plt.legend(['Real','Fake'], loc='upper left')
    plt.subplot(122)
    plt.plot(df['D_acc_real'])
    plt.plot(df['D_acc_fake'])
    #plt.plot(df['val_kld'])
    plt.title('Accuracy of Discriminator')
    plt.ylabel('acc')
    plt.xlabel('Training step')
    plt.legend(['Real','Fake'], loc='upper left')
    plt.savefig(os.path.join(save, 'fig3_2.jpg'))
    plt.close()
    
class ACGAN():
    def __init__(self):
        # Input shape
        self.img_rows = 28
        self.img_cols = 28
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 512
        self.num_classes = 10
        
        optimizer = Adam(0.0002, 0.5)
        losses = ['binary_crossentropy', 'sparse_categorical_crossentropy']

        # Build and compile the discriminator
        self.discriminator, self.generator = self.build_model()
        self.discriminator.compile(loss=losses,
            optimizer=optimizer,
            metrics=['accuracy'])

        # Build the generator

        # The generator takes noise and the target label as input
        # and generates the corresponding digit of that label
        noise = Input(shape=(self.latent_dim,))
        label = Input(shape=(1,))
        img = self.generator([noise, label])

        # For the combined model we will only train the generator
        self.discriminator.trainable = False

        # The discriminator takes generated image as input and determines validity
        # and the label of that image
        valid, target_label = self.discriminator(img)

        # The combined model  (stacked generator and discriminator)
        # Trains the generator to fool the discriminator
        self.combined = Model([noise, label], [valid, target_label])
        self.combined.compile(loss=losses,
            optimizer=optimizer)

    def build_model(self):
        image_size=64
        input_shape = (image_size, image_size, 3)
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
        validity = Dense(1, activation='sigmoid', name='discriminator_validity')(x)
        label = Dense(3, activation="softmax", name='discriminator_label')(x)
        discriminator = Model(inputs, [validity,label], name='discriminator')
        discriminator.summary()
        plot_model(discriminator, to_file='acgan_cnn_discriminator.png', show_shapes=True)

        # build generator model
        latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
        label_input = Input(shape=(1,), name='z_label')
        label_embedding = Flatten()(Embedding(1, latent_dim)(label_input))
        print('label_embedding')
        print(label_embedding.shape)
        merge = Concatenate()([latent_inputs, label_embedding])
        print('merge shape')
        print(merge.shape)
        x = Dense(shape[1] * shape[2] * shape[3], activation='relu')(merge)
        x = Reshape((shape[1], shape[2], shape[3]))(x)
        print('x shape')
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
        generator = Model([latent_inputs, label_input], outputs, name='generator')
        generator.summary()
        plot_model(generator, to_file='acgan_cnn_generator.png', show_shapes=True)
        
        return discriminator, generator
    

    def train(self, epochs, batch_size=128, sample_interval=10000, train_path='./hw4_data/train/', label_path='hw4_data/train.csv'):
        print(train_path)
        print(label_path)
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

        # Adversarial ground truths
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        # Load label of image
        df = pd.read_csv(label_path)
        y_train = df['Smiling']
        
        filename = './acgan1/train_history.csv'
        if not os.path.exists(os.path.join(filename)):
            f = open(os.path.join(filename),'w')
            w = csv.writer(f)
            w.writerow(['epoch', 'D_atribute_loss_real', 'D_atribute_loss_fake','D_acc_real','D_acc_fake','G_loss']) 
        
        for epoch in range(epochs):

            # ---------------------
            #  Train Discriminator
            # ---------------------

            # Select a random batch of images
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]

            # Sample noise as generator input
            noise = np.random.normal(0, 1, (batch_size, 512))

            # The labels of the digits that the generator tries to create an
            # image representation of
            sampled_labels = np.random.randint(0, 1, (batch_size, 1))

            # Generate a half batch of new images
            gen_imgs = self.generator.predict([noise, sampled_labels])

            # Image labels. 0-1 if image is valid or 2 if it is generated (fake)
            img_labels = y_train[idx]
            fake_labels = 2 * np.ones(img_labels.shape)

            # Train the discriminator
            self.discriminator.trainable = True
            self.discriminator.summary()
            d_loss_real = self.discriminator.train_on_batch(imgs, [valid, img_labels])
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, [fake, fake_labels])
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            print(d_loss)

            # ---------------------
            #  Train Generator
            # ---------------------
            self.discriminator.trainable = False
            # Train the generator
            self.combined.summary()
            g_loss = self.combined.train_on_batch([noise, sampled_labels], [valid, sampled_labels])
            w.writerows(np.array([epoch, d_loss_real[1], d_loss_fake[1], d_loss_real[3], d_loss_fake[3], g_loss[0]]).reshape(1,-1))
            # Plot the progress
            print ("%d [D loss: %f, acc.: %.2f%%, op_acc: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[3], 100*d_loss[4], g_loss[0]))

            # If at save interval => save generated image samples
            if epoch % sample_interval == 0:
                save = './acgan1'
                self.sample_images(epoch)
                model_name = 'model_'+str(epoch)+'.h5'
                self.generator.save(os.path.join(save, 'generator_'+model_name))
                self.discriminator.save(os.path.join(save, 'discriminator_'+model_name))
        f.close()
        model_name = 'model_'+str(epochs)+'.h5'
        self.generator.save(os.path.join('./acgan1', 'generator_'+model_name))
        self,discriminator.save(os.path.join('./acgan1', 'discriminator_'+model_name))

    def sample_images(self, epoch):
        r, c = 10, 10
        noise = np.random.normal(0, 1, (r * c, 512))
        sampled_labels = np.random.randint(0, 1, (r*c,))
        gen_imgs = self.generator.predict([noise, sampled_labels])

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt,:,:,:])
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("acgan_images1/%d.png" % epoch)
        plt.close()
        
generator = load_model('./generator_model_15000.h5')

directory = sys.argv[1]
save = sys.argv[2]

## fig3_2.jpg
plot_history(save)

## fig3_3.jpg
np.random.seed(1)
r, c = 2, 10
digit_size = 64
figure = np.zeros((digit_size * r, digit_size * c,3))
noise = np.random.normal(0, 1, (c, 512))
for i in range(r): 
    label = np.ones(10,)*i
    gen_imgs = generator.predict([noise,label])
    for j in range(c):
        img_tt = gen_imgs[j]
        figure[i * digit_size: (i + 1) * digit_size,
        j * digit_size: (j + 1) * digit_size,:] = img_tt
io.imsave(os.path.join(save,'fig3_3.jpg'), figure)