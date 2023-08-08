from tensorflow import keras
import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np

from tensorflow.keras.layers import Dense,LeakyReLU,Conv2D,Flatten,Reshape,Dropout,Conv2DTranspose,Input,Embedding,Concatenate
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.optimizers import Adam

def define_discriminator(in_shape=(28,28,1), n_classes=10):
    in_label = Input(shape=(1,))
    li = Embedding(n_classes, 50)(in_label)
    n_nodes = in_shape[0] * in_shape[1]
    li = Dense(n_nodes)(li)
    li = Reshape((in_shape[0], in_shape[1], 1))(li)

    in_image = Input(shape=in_shape)
    merge = Concatenate()([in_image, li])

    fe = Conv2D(128, (3,3), strides=(2,2), padding='same')(merge)
    fe = LeakyReLU(alpha=0.2)(fe)
    
    fe = Conv2D(128, (3,3), strides=(2,2), padding='same')(fe)
    fe = LeakyReLU(alpha=0.2)(fe)
    fe = Flatten()(fe)
    fe = Dropout(0.4)(fe)
    out_layer = Dense(1, activation='sigmoid')(fe)
    model = Model([in_image, in_label], out_layer)
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy'])
    return model

def define_generator(latent_dim, n_classes=10):
    in_label = Input(shape=(1,))
    li = Embedding(n_classes, 50)(in_label)
    n_nodes = 7 * 7
    li = Dense(n_nodes)(li)
    li = Reshape((7, 7, 1))(li)
    in_lat = Input(shape=(latent_dim,))
    n_nodes = 128 * 7 * 7
    gen = Dense(n_nodes)(in_lat)
    gen = LeakyReLU(alpha=0.2)(gen)
    gen = Reshape((7, 7, 128))(gen)
    merge = Concatenate()([gen, li])
    gen = Conv2DTranspose(128, (4,4), strides=(2,2), padding='same')(merge)
    gen = LeakyReLU(alpha=0.2)(gen)
    gen = Conv2DTranspose(128, (4,4), strides=(2,2), padding='same')(gen)
    gen = LeakyReLU(alpha=0.2)(gen)
    out_layer = Conv2D(1, (7,7), activation='tanh', padding='same')(gen)
    model = Model([in_lat, in_label], out_layer)
    return model

def define_gan(g_model, d_model):
    d_model.trainable = False
    gen_noise, gen_label = g_model.input
    gen_output = g_model.output
    gan_output = d_model([gen_output, gen_label])
    model = Model([gen_noise, gen_label], gan_output)
    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt)
    return model

def load_real_samples():
    (trainX, trainy), (_, _) =  keras.datasets.fashion_mnist.load_data()
    X = np.expand_dims(trainX, axis=-1)
    X = X.astype('float32')
    X = (X - 127.5) / 127.5
    return [X, trainy]

def generate_real_samples(dataset, n_samples):
    images, labels = dataset
    ix = np.random.randint(0, images.shape[0], n_samples)
    X, labels = images[ix], labels[ix]
    y = np.ones((n_samples, 1))
    return [X, labels], y

def generate_latent_points(latent_dim, n_samples, n_classes=10):
    x_input = np.random.randn(latent_dim * n_samples)
    z_input = x_input.reshape(n_samples, latent_dim)
    labels = np.random.randint(0, n_classes, n_samples)
    return [z_input, labels]

def generate_fake_samples(generator, latent_dim, n_samples):
    z_input, labels_input = generate_latent_points(latent_dim, n_samples)
    images = generator.predict([z_input, labels_input])
    y = np.zeros((n_samples, 1))
    return [images, labels_input], y

def train(g_model, d_model, gan_model, dataset, latent_dim, n_epochs=100, n_batch=128):
    bat_per_epo = int(dataset[0].shape[0] / n_batch)
    half_batch = int(n_batch / 2)
    for i in range(n_epochs):
        for j in range(bat_per_epo):
            [X_real, labels_real], y_real = generate_real_samples(dataset, half_batch)
            d_loss1, _ = d_model.train_on_batch([X_real, labels_real], y_real)
            [X_fake, labels], y_fake = generate_fake_samples(g_model, latent_dim, half_batch)
            d_loss2, _ = d_model.train_on_batch([X_fake, labels], y_fake)
            [z_input, labels_input] = generate_latent_points(latent_dim, n_batch)
            y_gan = np.ones((n_batch, 1))
            g_loss = gan_model.train_on_batch([z_input, labels_input], y_gan)
            if j % 200 == 0:
                print('>%d, %d/%d, d1=%.3f, d2=%.3f g=%.3f' %
                (i+1, j+1, bat_per_epo, d_loss1, d_loss2, g_loss))
    g_model.save('cgan_generator.h5')

if __name__ == "__main__":
    latent_dim = 100
    d_model = define_discriminator()
    g_model = define_generator(latent_dim)
    gan_model = define_gan(g_model, d_model)
    dataset = load_real_samples()
    train(g_model, d_model, gan_model, dataset, latent_dim)
    # 加载训练好的模型
    g_model.load_weights('../cgan_generator.h5')
    res = []
    labels = []
    for i in range(100):
        
        noise = np.random.randn(1,100)
        label = np.random.randint(0,9,1)
        labels.append(label)
        res.append(g_model.predict([noise,label]).squeeze())

    for i in range(100):
        plt.subplot(10, 10, 1 + i)
        plt.axis('off')
        plt.imshow(res[i], cmap='gray_r')
    plt.show()

    