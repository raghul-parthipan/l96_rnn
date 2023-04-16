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

def prepare_datasets_for_RNN(dataset,history_length):
    max_index = (dataset.shape[0]-1)//history_length
    dataset = dataset[:(max_index*history_length +1),:,:] 
    dataset_shape = dataset.shape[0]
    last_elements = dataset[-1,:,:]
    remaining_dataset = dataset[:-1,:,:]
    reshaped = remaining_dataset.reshape(-1,history_length,k,2)
    add_on = reshaped[1:,:1,:,:]
    last_elements = last_elements.reshape(1,1,k,2)
    add_on_combined = np.concatenate((add_on,last_elements),axis=0)
    concat = np.concatenate((reshaped,add_on_combined),axis=1)
    concat = concat.transpose((2,0,1,3)).reshape((-1,history_length+1,2),order="F")
    return concat.astype("float32")


if __name__ == "__main__":
    
    ######### USAGE ###########
    
    k = 8
    J = 32
    save_time_step = 0.005
    h=1 
    c=10
    b=10

    history_length = 100
    test_seq_length = 1000

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
    valid_dataset = np.load("../../data/truth_run/climate_change_exp/val.npy")


    u_t_train_1 = u_deriver(train1[:,:,0],train1[0,0,-1],save_time_step)
    u_t_train_2 = u_deriver(train2[:,:,0],train2[0,0,-1],save_time_step)
    u_t_train_3 = u_deriver(train3[:,:,0],train3[0,0,-1],save_time_step)
    u_t_train_4 = u_deriver(train4[:,:,0],train4[0,0,-1],save_time_step)
    u_valid = u_deriver(valid_dataset[:,:,0],valid_dataset[0,0,-1],save_time_step)

    x_combo = np.concatenate([train1[:-1,:,0],train2[:-1,:,0],train3[:-1,:,0],train4[:-1,:,0]],axis=0)

    u_combo = np.concatenate([u_t_train_1,u_t_train_2,u_t_train_3,u_t_train_4],axis=0)

    training_dataset = np.stack([x_combo,u_combo],axis=2)
    valid_dataset = np.stack([valid_dataset[:-1,:,0],u_valid],axis=2)

    train_nn_features = prepare_datasets_for_RNN(training_dataset,history_length)
    valid_nn_features = prepare_datasets_for_RNN(valid_dataset,test_seq_length)


    x_mean = np.mean(train_nn_features[:,:,0])
    x_std = np.std(train_nn_features[:,:,0])
    u_mean = np.mean(train_nn_features[:,:,1])
    u_std = np.std(train_nn_features[:,:,1])

    #scaling
    train_nn_features[:,:,0] = (train_nn_features[:,:,0] - x_mean)/x_std
    train_nn_features[:,:,1] = (train_nn_features[:,:,1] - u_mean)/u_std

    #scaling
    valid_nn_features[:,:,0] = (valid_nn_features[:,:,0] - x_mean)/x_std
    valid_nn_features[:,:,1] = (valid_nn_features[:,:,1] - u_mean)/u_std

    valid_nn_features_tf = tf.convert_to_tensor(valid_nn_features)

############################################       
#########  Model ###########
############################################

    h_shape = 34 # this is gan coding size

    ####################################################
############## BIDIRECTIONAL RNN ###################
####################################################
######to summarise the u sequence ##################

    hidden_state_size_bi = 32

    xu_seq = keras.layers.Input(shape=[None,2])

    layer1 = keras.layers.Bidirectional(keras.layers.GRU(hidden_state_size_bi,return_sequences=True))(inputs=xu_seq)
    layer2 = keras.layers.Bidirectional(keras.layers.GRU(hidden_state_size_bi,return_sequences=True))(inputs=layer1)
    layer3 = keras.layers.Bidirectional(keras.layers.GRU(hidden_state_size_bi,return_sequences=True))(inputs=layer2)
    output = keras.layers.TimeDistributed(keras.layers.Dense(1,bias_initializer="zeros"))(layer3)

    bi_rnn = keras.models.Model(inputs=xu_seq,outputs=output)

    class Sampling(keras.layers.Layer):
        def call(self, inputs):
            mean, log_var = inputs
            return K.random_normal(tf.shape(log_var)) * K.exp(log_var / 2) + mean
        
        #########################################################
    ################### H ENCODER ######################
    #########################################################

    encoder_hidden_state_size=32

    u_summary = keras.layers.Input(shape=[None,1])
    h_prev = keras.layers.Input(shape=[None,h_shape])
    x_in = keras.layers.Input(shape=[None,1])
    concat = keras.layers.Concatenate()([h_prev,u_summary,x_in])

    hidden_state_in = keras.layers.Input(shape=[encoder_hidden_state_size])
    hidden_state_in_2 = keras.layers.Input(shape=[encoder_hidden_state_size])

    output,state = keras.layers.GRU(encoder_hidden_state_size,return_sequences=True,return_state=True)(inputs=concat,initial_state=hidden_state_in)
    output2,state2 = keras.layers.GRU(encoder_hidden_state_size,return_sequences=True,return_state=True)(inputs=output,initial_state=hidden_state_in_2)

    h_mean = keras.layers.Dense(h_shape)(output2) + h_prev*0.7486
    h_log_var = keras.layers.Dense(h_shape)(output2)
    h_sample = Sampling()([h_mean,h_log_var])

    h_encoder = keras.models.Model(inputs=[h_prev,u_summary,x_in,hidden_state_in,hidden_state_in_2],outputs=[h_mean,h_log_var,h_sample,state,state2])


    #########################################################
    ################### First H step ######################
    #########################################################

    u_summary = keras.layers.Input(shape=[1])
    x_in = keras.layers.Input(shape=[1])

    concat = keras.layers.Concatenate()([u_summary,x_in])

    layer1 = keras.layers.Dense(16,activation="selu")(concat)
    layer2 = keras.layers.Dense(16,activation="selu")(layer1)
    layer3 = keras.layers.Dense(16,activation="selu")(layer2)


    h_mean = keras.layers.Dense(h_shape)(layer3) 
    h_log_var = keras.layers.Dense(h_shape)(layer3)

    h_mean = keras.layers.Reshape([1,h_shape])(h_mean)
    h_log_var = keras.layers.Reshape([1,h_shape])(h_log_var)
    h_sample = Sampling()([h_mean,h_log_var])


    h_encoder_first = keras.models.Model(inputs=[x_in,u_summary],outputs=[h_mean,h_log_var,h_sample])

    @tf.function
    def sample_from_encoder(xu_seq,encoder,first_encoder,encoder_hidden_state_size,bi_rnn):
        
        length = xu_seq.shape[1]
        batch_shape = xu_seq.shape[0]
        
        h_sequence = tf.TensorArray(dtype=tf.float32,size=0,dynamic_size=True)
        h_mean_out = tf.TensorArray(dtype=tf.float32,size=0,dynamic_size=True)
        h_log_var_out = tf.TensorArray(dtype=tf.float32,size=0,dynamic_size=True)
        
        u_summary = bi_rnn(xu_seq[:,:-1,:])

        h_mean1,h_log_var1,h_prev = first_encoder([xu_seq[:,0,0],u_summary[:,0,0]])
        h_sequence = h_sequence.write(0,h_prev)
        h_mean_out = h_mean_out.write(0,h_mean1)
        h_log_var_out = h_log_var_out.write(0,h_log_var1)

        
        hidden_state_1 = tf.zeros(shape=(batch_shape,encoder_hidden_state_size))
        hidden_state_2 = tf.zeros(shape=(batch_shape,encoder_hidden_state_size))    
        
        for n in tf.range(0,length-2):
            h_mean,h_log_var,h_sample,state,state2 = encoder([h_prev,u_summary[:,n+1:n+2,:],xu_seq[:,n+1:n+2,:1],
                                                        hidden_state_1,hidden_state_2])
            
            h_sequence = h_sequence.write(n+1,h_sample)
            h_prev = h_sample
            h_mean_out = h_mean_out.write(n+1,h_mean)
            h_log_var_out = h_log_var_out.write(n+1,h_log_var) 
            hidden_state_1 = state  
            hidden_state_2 = state2     
            
        h_sequence = h_sequence.stack()        
        h_mean_out_enc = h_mean_out.stack()
        h_log_var_out = h_log_var_out.stack()
        h_sequence = tf.transpose(h_sequence[:,:,0,:],[1,0,2])            
        h_mean_out_enc = tf.transpose(h_mean_out_enc[:,:,0,:],[1,0,2])
        h_log_var_out = tf.transpose(h_log_var_out[:,:,0,:],[1,0,2])

        return h_sequence,h_mean_out_enc,h_log_var_out
    
    generator = keras.models.load_model("gan_generator_final.h5")

    generator.trainable=False

    @tf.function
    def loglik_gaussian_u_cond_h(xu_seq,h_encoding,sigma):
        x_array =  xu_seq[:,:-1,:1]
        x_array_reshape = tf.reshape(x_array,(-1,1))
        u_array_reshape = tf.reshape(xu_seq[:,:-1,1:2],(-1,1))
        h_encoding_reshape = tf.reshape(h_encoding,(-1,h_shape))
        mean_u = generator([x_array_reshape,h_encoding_reshape])
        term = -K.log((sigma**2) *2*math.pi) - tf.math.divide((u_array_reshape-mean_u),sigma)**2
        loglik = 0.5*term
        return tf.reduce_mean(loglik) #average over t and k
        

    @tf.function
    def loglik_gaussian_h_encoder(h_encoding,h_mean,h_logvar):
        term1 = -(1/2)*(K.log(2*math.pi) + h_logvar)
        term2 = -((h_encoding-h_mean)**2)/(2*K.exp(h_logvar))
        loglik = term1+term2
        loglik = tf.reduce_sum(loglik,axis=[2]) #sum over the h dimensions 
        return tf.reduce_mean(loglik) #average over t and k 



    @tf.function
    def loglik_gaussian_h_gan(h_encoding):
        #### term 1 ####
        ### h drawn from normal(0,1) ####
        term = tf.reduce_sum(0.5*(-K.log((1**2) *2*math.pi) - tf.math.divide((h_encoding[:,:1,:]),1)**2),axis=2) #sum over z dimensions

        #### term 2 #####
        #### loglik for the rest of the markovian seq ####
        array = h_encoding[:,1:,:]
        phi = 0.7486
        mean = h_encoding[:,:-1,:]*phi
        sigma = (1-phi**2)**0.5
        
        term2 = 0.5*(-K.log((sigma**2) *2*math.pi) - tf.math.divide((array-mean),sigma)**2)
        term2 = tf.reduce_sum(term2,axis=[2]) #sum over the h dimensions 
        
        loglik_array = tf.concat([term,term2 ],axis=1)
        
        loglik = tf.reduce_mean(loglik_array) #average over t and batch_size (i.e. t and k) so that loglik is avg loglik per kt (what I do in other models)
                                                
        return loglik

    @tf.function
    def elbo_loss(loglik_u,loglik_h_gan,loglik_h_encoder):
        elbo = loglik_u + loglik_h_gan - loglik_h_encoder
        loss = -elbo
        return loss
    
    sigma_gan = 0.001

    class elbo_model(keras.models.Model):
            
        def __init__(self,gan_generator,h_encoder,h_encoder_first,bi_rnn,encoder_hidden_state_size,**kwargs):
            super().__init__(**kwargs) 
            self.gan_generator = gan_generator
            self.h_encoder = h_encoder
            self.h_encoder_first = h_encoder_first
            self.bi_rnn = bi_rnn
            self.encoder_hidden_state_size = encoder_hidden_state_size
            self.gan_generator.trainable = False
        
        def call(self,inputs,sigma_gan):
            """
            Inputs is just [training_nn_input] shape batch_size x history_length x 2
            """
            
            ##### Sample h sequences #####
            
            ### only 1 sample taken for importance sampling ###
            
            h_sequence,h_mean_out_enc,h_log_var_out = sample_from_encoder(inputs,self.h_encoder,self.h_encoder_first,
                                                                        self.encoder_hidden_state_size,self.bi_rnn)
            
            ### compute loglik and loss ###
            
            loglik_u = loglik_gaussian_u_cond_h(inputs,h_sequence,sigma_gan)
            loglik_h_gan = loglik_gaussian_h_gan(h_sequence)
            loglik_h_encoder = loglik_gaussian_h_encoder(h_sequence,h_mean_out_enc,h_log_var_out)
            loss = elbo_loss(loglik_u,loglik_h_gan,loglik_h_encoder)
            
            return loss
        
    model = elbo_model(gan_generator=generator,h_encoder=h_encoder,h_encoder_first=h_encoder_first,
                  bi_rnn=bi_rnn,encoder_hidden_state_size=encoder_hidden_state_size)
    

    ############ TRAIN MODEL ############

    optimizer=keras.optimizers.Adam(learning_rate=0.001)

    @tf.function
    def train_step(inputs):
        """Decorated train_step function which applies a gradient update to the parameters"""
        with tf.GradientTape() as tape:
            loss = model(inputs,sigma_gan,training=True)
            loss = tf.add_n([loss] + model.losses)
        gradients = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(gradients, model.trainable_variables))

        return loss

    def create_batch_tf(input_data,batch_s=32):
        shape_label = input_data.shape[0]
        batch_idx_la = np.random.choice(list(range(shape_label)), batch_s)
        batch_item = input_data[batch_idx_la,]
        return tf.convert_to_tensor(batch_item)

    def fit_model(input_list,epochs,model,history,validation_loss,valid_list,batch_size=32):
        
        start = time.time()
        K.clear_session()
        batch_loss = []
        batches_per_epoch = int(np.floor(input_list.shape[0]/batch_size))
        
        for epoch in tf.range(epochs):            
            for i in range(batches_per_epoch):

                batch_list= create_batch_tf(input_list,batch_size)
                loss = train_step(batch_list)
                batch_loss.append(loss)
                
            training_loss_for_epoch = list_average(batch_loss)
            batch_loss = []
            history.append(training_loss_for_epoch)
            
            val_loss = model(valid_list,sigma_gan)
            validation_loss.append(val_loss)
            


            h_encoder.save("h_encoder_final.h5")
            bi_rnn.save("bi_rnn_final.h5")
            h_encoder_first.save("h_encoder_first_final.h5")
            
        return history, validation_loss
    
    tf.config.run_functions_eagerly(False)

    input_history = []

    valid_history = []

    fit_model(train_nn_features[:],20,model,input_history,valid_history,valid_nn_features_tf,batch_size=64)