#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import sklearn
import time
from pickle import load
import tensorflow as tf
from tensorflow import keras

K = keras.backend

from sklearn.preprocessing import StandardScaler,PowerTransformer
import math
from scipy.stats import multivariate_normal
from scipy.stats import norm
import os
import pickle
from helper import *

def u_deriver(x_train,F,save_time_step):
    u_t = -np.roll(x_train[:-1,:],1,axis=1) * (np.roll(x_train[:-1,:],2,axis=1) - np.roll(x_train[:-1,:],-1,axis=1)) - x_train[:-1,:] + F -(x_train[1:,:] - x_train[:-1,:])/save_time_step
    return u_t

def create_batch(input_list, batch_s=32):
    
    batch_list = []
    shape_label = input_list[0].shape[0]
    batch_idx_la = np.random.choice(list(range(shape_label)), batch_s)
    for i in input_list: 
        batch_item = (i[batch_idx_la,])
        batch_list.append(batch_item)
    
    del batch_idx_la
            
    return batch_list

def list_average(list_of_loss):
    return sum(list_of_loss)/len(list_of_loss)




if __name__ == "__main__":
    
    ######### USAGE ###########
    
    k = 8
    J = 32
    save_time_step = 0.005
    h=1 
    c=10
    b=10

    batch_size=1024
    epochs=30
    # epochs = 10

    seed = 42

    tf.random.set_seed(seed)
    np.random.seed(seed)



     ###############################
    ########### LOAD DATA ##############
    ###############################

    train1 = np.load("../../data/truth_run/climate_change_exp/train_set_1.npy")
    train2 = np.load("../../data/truth_run/climate_change_exp/train_set_2.npy")
    train3 = np.load("../../data/truth_run/climate_change_exp/train_set_3.npy")
    f20 = np.load("../../data/truth_run/training_dataset.npy")
    train4 = np.concatenate([f20, 20*np.ones(shape=(f20.shape[0],k,1))],axis=2)[:200000]

    u_t_train_1 = u_deriver(train1[:,:,0],train1[0,0,-1],save_time_step)
    u_t_train_2 = u_deriver(train2[:,:,0],train2[0,0,-1],save_time_step)
    u_t_train_3 = u_deriver(train3[:,:,0],train3[0,0,-1],save_time_step)
    u_t_train_4 = u_deriver(train4[:,:,0],train4[0,0,-1],save_time_step)

    x_combo = np.ravel(np.concatenate([train1[:-1,:,0],train2[:-1,:,0],train3[:-1,:,0],train4[:-1,:,0]],axis=0)).reshape(-1,1)

    u_combo = np.ravel(np.concatenate([u_t_train_1,u_t_train_2,u_t_train_3,u_t_train_4],axis=0)).reshape(-1,1)

    scaler1 = StandardScaler()
    scaler2 = StandardScaler()

    x_t_scaled = scaler1.fit_transform(x_combo)
    u_t_scaled = scaler2.fit_transform(u_combo)

    input_dataset = [x_t_scaled,u_t_scaled]

############################################       
#########  Model X-sml-r  ###########
############################################

    """
    Generator.

    Called "generator_dense_stoch" in the gagne github. The relevant files are:
    train_loren_gan.py - def train_lorenz_gan
    The config file used is lorenz_gan_travis.yaml
    gan structure is "specified random" which from the "def train_lorenz_gan" means generator is the generator_dense_stoch 
    and discriminator is discriminator_dense, both of which are specified in gan.py (the model layers are specified) as
    separate functions
    """

    codings_size = 1 + 1 + 2*16
    noise_sd_training = 0.01

    x_in = keras.layers.Input(shape=[1])
    z_in = keras.layers.Input(shape=[codings_size])

    z_in_input = z_in[:,:1]
    z_in_add_input = z_in[:,1:2]
    z_in_dense1 = z_in[:,2:2+16]
    z_in_dense2 = z_in[:,-16:]

    z_in_add_input_scaled = z_in_add_input*noise_sd_training
    z_in_dense1_scaled = z_in_dense1*noise_sd_training
    z_in_dense2_scaled = z_in_dense2*noise_sd_training

    noisy_x = x_in + z_in_add_input_scaled
    concat = keras.layers.Concatenate()([noisy_x,z_in_input])

    layer1 = keras.layers.Dense(16,activation="selu",kernel_regularizer=keras.regularizers.l2(0.001))(concat)
    layer1_noise = layer1 + z_in_dense1_scaled

    layer2 = keras.layers.Dense(16,activation="selu",kernel_regularizer=keras.regularizers.l2(0.001))(layer1_noise)
    layer2_noise = layer2 + z_in_dense2_scaled

    final = keras.layers.Dense(1,kernel_regularizer=keras.regularizers.l2(0.001))(layer2_noise)
    final_normalised = keras.layers.BatchNormalization()(final)

    generator = keras.models.Model(inputs=[x_in,z_in],outputs=[final_normalised])  




    """
    Discriminator.

    Called "discriminator_dense" in Gagne github. See notes from generator above.
    """

    u_in = keras.layers.Input(shape=[1])
    x_in = keras.layers.Input(shape=[1])
    concat = keras.layers.Concatenate()([x_in,u_in])
    add_noise_1 = keras.layers.GaussianNoise(noise_sd_training)(concat) #these noise layers are only used in training
    layer1 = keras.layers.Dense(16,activation="selu",kernel_regularizer=keras.regularizers.l2(0.001))(add_noise_1)
    add_noise_2 = keras.layers.GaussianNoise(noise_sd_training)(layer1)
    layer2 = keras.layers.Dense(16,activation="selu",kernel_regularizer=keras.regularizers.l2(0.001))(add_noise_2)
    final = keras.layers.Dense(1,activation="sigmoid",kernel_regularizer=keras.regularizers.l2(0.001))(layer2)

    discriminator = keras.models.Model(inputs=[x_in,u_in],outputs=[final])


    # generator = keras.models.load_model("gan_generator_updatedv2.h5")
    # discriminator = keras.models.load_model("gan_discriminator_updatedv2.h5")


    #important to include this epsilon to prevent overflow and nans.
    def discriminator_loss(real_data,fake_data):
        epsilon = 1e-7
        epsilon_ = tf.constant(epsilon)
        real_data_pred = tf.clip_by_value(discriminator(real_data),epsilon_,1. - epsilon_)
        term1 = 0.5*(K.log(real_data_pred + epsilon_))
        fake_data_pred = tf.clip_by_value(discriminator(fake_data),epsilon_,1. - epsilon_)
        term2 = 0.5*(K.log(1.-fake_data_pred+ epsilon_))
        loglik = term1+term2
        loss = -tf.reduce_mean(loglik)
        return loss

    #non saturating 
    def generator_loss(fake_data):
        epsilon = 1e-7
        epsilon_ = tf.constant(epsilon)   
        fake_data_pred = tf.clip_by_value(discriminator(fake_data),epsilon_,1. - epsilon_)
        term = 0.5*(K.log(fake_data_pred + epsilon_))
        loss = -tf.reduce_mean(term)
        return loss

    d_optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001)
    g_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

    @tf.function
    def disc_step(real_data,fake_data):
        """Decorated train_step function which applies a gradient update to the parameters"""
        with tf.GradientTape() as tape:
            loss = discriminator_loss(real_data,fake_data)
            loss = tf.add_n([loss] + discriminator.losses)
        gradients = tape.gradient(loss, discriminator.trainable_variables)
        d_optimizer.apply_gradients(zip(gradients, discriminator.trainable_variables))
        return loss

    @tf.function 
    def gen_step(noise,real_data_x):
        """Decorated train_step function which applies a gradient update to the parameters"""
        with tf.GradientTape() as tape:
            fake_u = generator([real_data_x,noise])
            fake_data = [real_data_x,fake_u]
            loss = generator_loss(fake_data)
            loss = tf.add_n([loss] + generator.losses)
        gradients = tape.gradient(loss, generator.trainable_variables)
        g_optimizer.apply_gradients(zip(gradients, generator.trainable_variables))
        return loss


    K.clear_session()    
    batches_per_epoch = int(np.floor(input_dataset[0].shape[0]/batch_size))
    gen_batch_loss = []
    gen_history_full = []   
    half_batch = int(batch_size/2)
    for epoch in range(epochs):

        for i in range(batches_per_epoch):

            batch_list= create_batch(
                input_dataset,batch_size)
            x_real = batch_list[0][:half_batch]
            u_real = batch_list[1][:half_batch]
            real_data = [x_real,u_real]                    
            random_noise = tf.random.normal(shape=(half_batch,codings_size))
            x_real_2 = batch_list[0][half_batch:]
            u_fake = generator([x_real_2,random_noise])
            fake_data = [x_real_2,u_fake]
            loss = disc_step(real_data,fake_data)

            batch_list= create_batch(
                input_dataset,batch_size)
            random_noise = tf.random.normal(shape=(batch_size,codings_size))
            x_real = batch_list[0]
            loss = gen_step(random_noise,x_real)
            gen_batch_loss.append(loss)
            gen_average_batch_loss = list_average(gen_batch_loss)
        
        gen_batch_loss = []
        gen_history_full.append(gen_average_batch_loss)

        np.save("generator_losses.npy",np.array(gen_history_full))
        generator.save("test_generator_2.h5")
        discriminator.save("test_discriminator_2.h5")

    generator.save("gan_generator_final.h5")
    discriminator.save("gan_discriminator_final.h5")

    