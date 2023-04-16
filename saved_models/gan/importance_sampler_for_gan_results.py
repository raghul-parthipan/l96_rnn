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

def _rhs_X_dt(X, F,U,dt=0.005):
    """Compute the right hand side of the X-ODE."""

    dXdt = (-np.roll(X, 1,axis=1) * (np.roll(X, 2,axis=1) - np.roll(X, -1,axis=1)) -
                X + F - U)

    return dt * dXdt 


def U(Xt,Xt_1,F,dt=0.005):
    k1_X = _rhs_X_dt(Xt,F,U=0)
    k2_X = _rhs_X_dt(Xt + k1_X / 2,F, U=0)
    Xt_1_pred = k2_X + Xt 
    #print(Xt_1_pred)
    Ut = (Xt_1_pred - Xt_1 )/dt

    return Ut


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
    last_elements = dataset[-1,:,:]
    remaining_dataset = dataset[:-1,:,:]
    reshaped = remaining_dataset.reshape(-1,history_length,k,2)
    add_on = reshaped[1:,:1,:,:]
    last_elements = last_elements.reshape(1,1,k,2)
    add_on_combined = np.concatenate((add_on,last_elements),axis=0)
    concat = np.concatenate((reshaped,add_on_combined),axis=1)
    concat = concat.transpose((2,0,1,3)).reshape((-1,history_length+1,2),order="F")
    return concat.astype("float32")

def prep_for_loglik_estimation(dataset,history_length,exp_its,conf_its):
    dataset = dataset.astype("float32")
    max_index = (dataset.shape[0]-1)//history_length
    dataset = dataset[:(max_index*history_length +1),:,:] 
    last_elements = dataset[-1,:,:]
    remaining_dataset = dataset[:-1,:,:]
    reshaped = remaining_dataset.reshape(-1,history_length,k,2)
    add_on = reshaped[1:,:1,:,:]
    last_elements = last_elements.reshape(1,1,k,2)
    add_on_combined = np.concatenate((add_on,last_elements),axis=0)
    concat = np.concatenate((reshaped,add_on_combined),axis=1)
    new = np.repeat(concat.transpose((0,2,1,3)),exp_its,axis=0).reshape(-1,history_length+1,2)
    new = np.tile(new,(conf_its,1,1))
    return new.astype("float32")


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

    test_dataset = np.load("../../data/truth_run/climate_eval_dataset.npy")
    test_dataset_28 = np.load( "../../data/truth_run/climate_change_exp/full_28_set.npy")

    u_t_train_1 = U(train1[:-1,:,0],train1[1:,:,0],train1[0,0,-1])
    u_t_train_2 = U(train2[:-1,:,0],train2[1:,:,0],train2[0,0,-1])
    u_t_train_3 = U(train3[:-1,:,0],train3[1:,:,0],train3[0,0,-1])
    u_t_train_4 = U(train4[:-1,:,0],train4[1:,:,0],train4[0,0,-1])

    u_t_train = np.concatenate([u_t_train_1,u_t_train_2,u_t_train_3,u_t_train_4],axis=0)
    x_combo = np.concatenate([train1[:-1,:,0],train2[:-1,:,0],train3[:-1,:,0],train4[:-1,:,0]],axis=0)

    u_t_test = U(test_dataset[:-1,:,0],test_dataset[1:,:,0],test_dataset[0,0,-1])
    x_test = test_dataset[:-1,:,0]

    u_t_test_28 = U(test_dataset_28[:-1,:,0],test_dataset_28[1:,:,0],test_dataset_28[0,0,-1])
    x_test_28 = test_dataset_28[:-1,:,0]


    training_dataset = np.stack([x_combo,u_t_train],axis=2)
    test_dataset = np.stack([x_test,u_t_test],axis=2)
    test_dataset_28 = np.stack([x_test_28,u_t_test_28],axis=2)

    train_nn_features = prepare_datasets_for_RNN(training_dataset,history_length)
    test_for_loglik = prep_for_loglik_estimation(test_dataset[:],6000,1,1)
    test_for_loglik_28 = prep_for_loglik_estimation(test_dataset_28[:],6000,1,1)


    x_mean = np.mean(train_nn_features[:,:,0])
    x_std = np.std(train_nn_features[:,:,0])
    u_mean = np.mean(train_nn_features[:,:,1])
    u_std = np.std(train_nn_features[:,:,1])


    #scaling
    train_nn_features[:,:,0] = (train_nn_features[:,:,0] - x_mean)/x_std
    train_nn_features[:,:,1] = (train_nn_features[:,:,1] - u_mean)/u_std

    #scaling
    test_for_loglik_28[:,:,0] = (test_for_loglik_28[:,:,0] - x_mean)/x_std
    test_for_loglik_28[:,:,1] = (test_for_loglik_28[:,:,1] - u_mean)/u_std      

    test_for_loglik_28_tf = tf.convert_to_tensor(test_for_loglik_28)
    test_for_loglik_tf = tf.convert_to_tensor(test_for_loglik)

############################################       
#########  Model ###########
############################################

    class Sampling(keras.layers.Layer):
        def call(self, inputs):
            mean, log_var = inputs
            return K.random_normal(tf.shape(log_var)) * K.exp(log_var / 2) + mean
        
    z_shape = 34
    hidden_state_size_bi = 32
    encoder_hidden_state_size=32

    generator = keras.models.load_model("gan_generator_updatedv2_b.h5")

    h_encoder = keras.models.load_model("h_encoder_test.h5",custom_objects={
            "Sampling":Sampling})
    bi_rnn = keras.models.load_model("bi_rnn_test.h5")
    h_encoder_first = keras.models.load_model("h_encoder_first_test.h5",custom_objects={
            "Sampling":Sampling})
    
    def sample_from_encoder(xu_seq,encoder,first_encoder,encoder_hidden_state_size,bi_rnn):
    
        length = xu_seq.shape[1]
        batch_shape = xu_seq.shape[0]
        
        h_sequence = tf.TensorArray(dtype=tf.float32,size=0,dynamic_size=True)
        h_mean_out = tf.TensorArray(dtype=tf.float32,size=0,dynamic_size=True)
        h_log_var_out = tf.TensorArray(dtype=tf.float32,size=0,dynamic_size=True)
        
        u_summary = bi_rnn.predict(xu_seq[:,:-1,:])

        h_mean1,h_log_var1,h_prev = first_encoder.predict([xu_seq[:,0,0],u_summary[:,0,0]])
        h_sequence = h_sequence.write(0,h_prev)
        h_mean_out = h_mean_out.write(0,h_mean1)
        h_log_var_out = h_log_var_out.write(0,h_log_var1)

        
        hidden_state_1 = tf.zeros(shape=(batch_shape,encoder_hidden_state_size))
        hidden_state_2 = tf.zeros(shape=(batch_shape,encoder_hidden_state_size))    
        
        for n in tf.range(0,length-2):
            h_mean,h_log_var,h_sample,state,state2 = encoder.predict([h_prev,u_summary[:,n+1:n+2,:],xu_seq[:,n+1:n+2,:1],
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
            
    
    ######## Loglik estimations ########

    def eval_loglik_gaussian_u_cond_h(xu_seq,h_encoding,sigma):
        x_array =  xu_seq[:,:-1,:1]
        x_array_reshape = tf.reshape(x_array,(-1,1))
        u_array_reshape = tf.reshape(xu_seq[:,:-1,1:2],(-1,1))
        h_encoding_reshape = tf.reshape(h_encoding,(-1,z_shape))
        mean_u = generator([x_array_reshape,h_encoding_reshape])
        term = -K.log((sigma**2) *2*math.pi) - tf.math.divide((u_array_reshape-mean_u),sigma)**2
        loglik = 0.5*term
        loglik = tf.reshape(loglik,(-1,xu_seq.shape[1]-1,1))
        return tf.reduce_sum(loglik,axis=1) #take sum over time
        
    def eval_loglik_gaussian_h_encoder(h_encoding,h_mean,h_logvar):
        term1 = -(1/2)*(K.log(2*math.pi) + h_logvar)
        term2 = -((h_encoding-h_mean)**2)/(2*K.exp(h_logvar))
        loglik = term1+term2
        loglik = tf.reduce_sum(loglik,axis=[2],keepdims=True) #sum over the h dimensions 
        return tf.reduce_sum(loglik,axis=1) #sum over time 

    def eval_loglik_gaussian_h_gan(h_encoding):
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
        loglik = tf.reduce_sum(loglik_array,axis=1,keepdims=True) #sum over t
                            
        return loglik
    
    def loglik_exact_approx(testing_sequence,sigma,it):

        """
        
        it is number for approx
        
        """
        
        combined = tf.zeros(shape=(int(testing_sequence.shape[0]/k),1))
        
        print("expectation_outer_loop")
        
        for n in tf.range(it):
            
            print(n)
        
            h_sequence,h_mean_out_enc,z_log_var_out = sample_from_encoder(testing_sequence,h_encoder,h_encoder_first,
                                                        encoder_hidden_state_size,bi_rnn)

            print("generated")
            exponent = eval_loglik_gaussian_u_cond_h(testing_sequence,h_sequence,sigma) +  eval_loglik_gaussian_h_gan(h_sequence) \
            - eval_loglik_gaussian_h_encoder(h_sequence,h_mean_out_enc,z_log_var_out)

            reshape_exponent = tf.reshape(exponent,(-1,k))
            exponent_summed = tf.reduce_sum(reshape_exponent,axis=1)
            
            reshaped = tf.reshape(exponent_summed,(-1,1)) 
            
            combined = tf.concat([combined,reshaped],axis=1)
            
        result = combined[:,1:]
        
        most_positive_alpha = np.max(result,axis=1)
        remaining_factor = result[:,:] - most_positive_alpha.reshape(-1,1)[:,:]
        second_positive = np.sort(remaining_factor,axis=1)[:,-2]

        x = np.exp(second_positive.astype("float64"))
        term3 = x - (x**2)/2    
        
        loglik = -np.log(float(result.shape[1])) + most_positive_alpha +term3
        
        avg_loglik = np.sum(loglik) / ((testing_sequence.shape[0])*(testing_sequence.shape[1]-1))
                
        return avg_loglik



    def confidence_interval_arrays(num_its_conf,testing_sequence,sigma,it_mc):
        
        #testing_sequence must have testing_sequence.shape[0] be a multiple of K
        
        array = np.zeros(shape=(num_its_conf,1))
        
        for i in tf.range(num_its_conf):
            
            print("confidence iteration",i)
            
            loglik = loglik_exact_approx(testing_sequence,sigma,it_mc)
            array[i,0] = loglik
        return array
    

    def rx_star(array):
        return np.random.choice(array,size=len(array))


    ####### F = 20 #########

    a = confidence_interval_arrays(5,test_for_loglik_tf,0.001,5)
    np.save("loglik_20.npy",a)

    a = np.load("loglik_20.npy")



    dt = 0.005

    a_with_correction = a -np.log(dt*u_std)
    a_sample = [np.mean(rx_star(a_with_correction.reshape(-1))) for _ in range(10000)]
    print(np.mean(a_with_correction))

    lo, hi = np.quantile(a_sample,[0.025,0.975])
    print(lo)
    print(hi)

    ####### F = 28 ########

    a = confidence_interval_arrays(5,test_for_loglik_28_tf,0.001,5)
    np.save("loglik_28.npy",a)
    
    a = np.load("loglik_28.npy")

    dt = 0.005

    a_with_correction = a -np.log(dt*u_std)

    a_sample = [np.mean(rx_star(a_with_correction.reshape(-1))) for _ in range(10000)]
    print(np.mean(a_with_correction))

    lo, hi = np.quantile(a_sample,[0.025,0.975])
    print(lo)
    print(hi)
