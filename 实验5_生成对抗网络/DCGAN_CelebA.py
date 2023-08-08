import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import random
from load_celeba10000 import load_data


latent_dim = 512

def make_generator_model(latent_dim):
    leakyrelu = layers.LeakyReLU()
    latent_inputs = keras.Input(shape=(latent_dim,))
    x = layers.Dense(32 * 32 * 1, activation="relu")(latent_inputs)
    x = layers.Reshape((32, 32, 1))(x)
    x = layers.Conv2DTranspose(512, 4, activation="relu", strides=1, padding="same")(x)
    x = layers.Conv2DTranspose(256, 3, activation="relu", strides=2, padding="same")(x)
    generator_outputs = layers.Conv2DTranspose(3,3,activation="sigmoid", padding="same")(x)
    generator = keras.Model(latent_inputs, generator_outputs, name="generator")
    #generator.summary()
    return generator

def make_discriminator_model():
    
    encoder_inputs = keras.Input(shape=(64, 64, 3))
    x = layers.Conv2D(filters = 256, kernel_size = 3, activation="relu", strides=2, padding="same")(encoder_inputs)
    x = layers.Conv2D(512, 3, activation="relu", strides=2, padding="same")(x)
    x = layers.Conv2D(128, 3, activation="relu", strides=2, padding="same")(x)
    x = layers.Flatten()(x)
    x = layers.Dense(16, activation="relu")(x)
    discriminator_outputs = layers.Dense(1, activation = "sigmoid")(x)
    
    discriminator = keras.Model(encoder_inputs, discriminator_outputs, name="discriminator")
    opt = keras.optimizers.Adam(lr=0.0002, beta_1=0.5)
    discriminator.compile(loss='binary_crossentropy', optimizer=opt)
    return discriminator

def define_gan(g_model, d_model):
    d_model.trainable = False
    gen_noise = g_model.input
    gen_output = g_model.output
    gan_output = d_model(gen_output)
    model = keras.Model(gen_noise, gan_output)
    opt = keras.optimizers.Adam(lr=0.0005, beta_1=0.5)
    model.compile(loss='binary_crossentropy', optimizer=opt)
    return model

def train(epochs):

    for epoch in range(epochs):
        print("Epoch is", epoch)
        for step, image_batch in enumerate(ds):
            
            BATCH_SIZE = image_batch.shape[0]

            noise = tf.random.normal((BATCH_SIZE, latent_dim))
            generated_images = g(noise)
            
            combined_images = tf.concat([generated_images, image_batch], axis=0)
            labels = tf.concat([tf.zeros((BATCH_SIZE, 1)), tf.ones((BATCH_SIZE, 1))], axis=0)
            
            d_loss = d.train_on_batch(combined_images, labels)
            
            g_loss = dcgan.train_on_batch(noise, tf.ones((BATCH_SIZE,1)))
            
            if step % 30 == 0:
                print("batch: [%d/%d]: d_loss : %.5f, g_loss : %.5f" % (step,BATCH_SIZE, d_loss, g_loss))

if __name__ == "__main__":
    ds = load_data(128)
    d = make_discriminator_model()
    g = make_generator_model(latent_dim)
    dcgan = define_gan(g,d)
    with tf.device('gpu'):
        train(1)
    g.save('dcgan_generator.h5')
    d.save('dcgan_discriminator.h5')
    
                