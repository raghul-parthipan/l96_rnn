#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import sklearn
import time
from pickle import load

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,PowerTransformer
import math
from scipy.stats import multivariate_normal
from scipy.stats import norm
from statsmodels.tsa.stattools import acf
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import tensorflow as tf
from tensorflow import keras


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


if __name__ == "__main__":

    k = 8
    J = 32

    
    save_time_step = 0.005
    h=1 
    c=10
    b=10

    nlags = 1000

###################
####### load data #######
###################

    truth_set_full = np.load("../../data/truth_run/f_20_all.npy")
    truth_set_shorter = np.load("../../data/truth_run/climate_eval_dataset.npy")[:,:,0]

    poly_full = np.load("../../data/simulation_runs/polynomial_param/new_generated_data_20.npy").transpose([1,0])

    rnn_full = np.load("../../data/simulation_runs/rnn/rnn_final/f_20_data/f_20_all.npy")

    gan_full = np.load("../../data/simulation_runs/gan_gagne/corrected_gans/new_generated_data_20.npy").transpose([1,0])

    truth_set_full_28  = np.load("../../data/truth_run/climate_change_exp/full_28_set.npy")[:,:,0]

    gan_full_28 = (np.load("../../data/simulation_runs/gan_gagne/corrected_gans/new_generated_data_28.npy")).transpose([1,0])
    rnn_full_28 =(np.load("../../data/simulation_runs/rnn/rnn_final/clim_change/f_28_all.npy"))

##################
###### X acf #######
####################

    truth_acf_full = np.ones((8,1000))

    for i in range(8):
        truth_acf = acf(truth_set_full[:1000000,i],adjusted=True,nlags=nlags-1)
        truth_acf_full[i,:] = truth_acf

    poly_acf_full  = np.ones((8,1000))

    for i in range(8):

        poly_acf = acf(poly_full[:1000000,i],adjusted=True,nlags=nlags-1)
        poly_acf_full[i,:] = poly_acf

    rnn_acf_full  = np.ones((8,1000))

    for i in range(8):

        rnn_acf = acf(rnn_full[:1000000,i],adjusted=True,nlags=nlags-1)
        rnn_acf_full[i,:] = rnn_acf

    gan_acf_full = np.ones((8,1000))

    for i in range(8):

        gan_acf = acf(gan_full[:1000000,i],adjusted=True,nlags=nlags-1)
        gan_acf_full[i,:] = gan_acf




    np.save("acf_data/truth_acf_full.npy",truth_acf_full)
    np.save("acf_data/poly_acf_full.npy",poly_acf_full)
    np.save("acf_data/rnn_acf_full.npy",rnn_acf_full)
    np.save("acf_data/gan_acf_full.npy",gan_acf_full)

    truth_acf_full_28 = np.ones((8,1000))

    for i in range(8):
        truth_acf = acf(truth_set_full_28[:1000000,i],adjusted=True,nlags=nlags-1)
        truth_acf_full_28[i,:] = truth_acf

    rnn_acf_full_28  = np.ones((8,1000))

    for i in range(8):

        rnn_acf = acf(rnn_full_28[:1000000,i],adjusted=True,nlags=nlags-1)
        rnn_acf_full_28[i,:] = rnn_acf

    gan_acf_full_28 = np.ones((8,1000))

    for i in range(8):

        gan_acf = acf(gan_full_28[:1000000,i],adjusted=True,nlags=nlags-1)
        gan_acf_full_28[i,:] = gan_acf

    np.save("acf_data/truth_acf_full_28.npy",truth_acf_full_28)
    np.save("acf_data/rnn_acf_full_28.npy",rnn_acf_full_28)
    np.save("acf_data/gan_acf_full_28.npy",gan_acf_full_28)

#####################
######### sub grid forcing acf ########
#########################

    U_true = U(truth_set_full[:-1,:],truth_set_full[1:,:],F=20)
    U_true_shorter = U(truth_set_shorter[:-1,:],truth_set_shorter[1:,:],F=20)

    U_rnn =  U(rnn_full[:-1,:],rnn_full[1:,:],F=20)
    U_gan = U(gan_full[:-1:,:],gan_full[1:,:],F=20)
    U_poly = U(poly_full[:-1,:],poly_full[1:,:],F=20)

    truth_acf_full = np.ones((8,1000))

    for i in range(8):
        truth_acf = acf(U_true[:1000000,i],adjusted=True,nlags=nlags-1)
        truth_acf_full[i,:] = truth_acf

    poly_acf_full  = np.ones((8,1000))

    for i in range(8):

        poly_acf = acf(U_poly[:1000000,i],adjusted=True,nlags=nlags-1)
        poly_acf_full[i,:] = poly_acf

    rnn_acf_full  = np.ones((8,1000))

    for i in range(8):

        rnn_acf = acf(U_rnn[:1000000,i],adjusted=True,nlags=nlags-1)
        rnn_acf_full[i,:] = rnn_acf

    gan_acf_full = np.ones((8,1000))

    for i in range(8):

        gan_acf = acf(U_gan[:1000000,i],adjusted=True,nlags=nlags-1)
        gan_acf_full[i,:] = gan_acf

    np.save("acf_data/truth_acf_u.npy",truth_acf_full)
    np.save("acf_data/poly_acf_u.npy",poly_acf_full)
    np.save("acf_data/rnn_acf_u.npy",rnn_acf_full)
    np.save("acf_data/gan_acf_u.npy",gan_acf_full)

    U_true_28 = U(truth_set_full_28[:-1,:],truth_set_full_28[1:,:],F=28)
    U_rnn_28 =  U(rnn_full_28[:-1,:],rnn_full_28[1:,:],F=28)
    U_gan_28 = U(gan_full_28[:-1,:],gan_full_28[1:,:],F=28)

    truth_acf_full_28 = np.ones((8,1000))

    for i in range(8):
        truth_acf = acf(U_true_28[:1000000,i],adjusted=True,nlags=nlags-1)
        truth_acf_full_28[i,:] = truth_acf

    rnn_acf_full_28  = np.ones((8,1000))

    for i in range(8):

        rnn_acf = acf(U_rnn_28[:1000000,i],adjusted=True,nlags=nlags-1)
        rnn_acf_full_28[i,:] = rnn_acf

    gan_acf_full_28 = np.ones((8,1000))

    for i in range(8):

        gan_acf = acf(U_gan_28[:1000000,i],adjusted=True,nlags=nlags-1)
        gan_acf_full_28[i,:] = gan_acf

    np.save("acf_data/truth_acf_u_28.npy",truth_acf_full_28)
    np.save("acf_data/rnn_acf_u_28.npy",rnn_acf_full_28)
    np.save("acf_data/gan_acf_u_28.npy",gan_acf_full_28)


##############
##### residual acf #######
################

    # polynomial

   

    sigma=2.01156
    phi=0.985728

    train1 = np.load("../../data/truth_run/climate_change_exp/train_set_1.npy")
    train2 = np.load("../../data/truth_run/climate_change_exp/train_set_2.npy")
    train3 = np.load("../../data/truth_run/climate_change_exp/train_set_3.npy")
    f20 = np.load("../../data/truth_run/training_dataset.npy")
    train4 = np.concatenate([f20, 20*np.ones(shape=(f20.shape[0],k,1))],axis=2)[:200000]

    def u_deriver(x_train,F,save_time_step=save_time_step):
        u_t = -np.roll(x_train[:-1,:],1,axis=1) * (np.roll(x_train[:-1,:],2,axis=1) - np.roll(x_train[:-1,:],-1,axis=1)) - x_train[:-1,:] + F -(x_train[1:,:] - x_train[:-1,:])/save_time_step
        return u_t

    u_t_train_1 = u_deriver(train1[:,:,0],train1[0,0,-1])
    u_t_train_2 = u_deriver(train2[:,:,0],train2[0,0,-1])
    u_t_train_3 = u_deriver(train3[:,:,0],train3[0,0,-1])
    u_t_train_4 = u_deriver(train4[:,:,0],train4[0,0,-1])

    x_combo = np.ravel(np.concatenate([train1[:-1,:,0],train2[:-1,:,0],train3[:-1,:,0],train4[:-1,:,0]],axis=0)).reshape(-1,1)


    u_combo = np.ravel(np.concatenate([u_t_train_1,u_t_train_2,u_t_train_3,u_t_train_4],axis=0)).reshape(-1,1)

    u_combo_no_ravel = np.concatenate([u_t_train_1,u_t_train_2,u_t_train_3,u_t_train_4],axis=0)

    poly = PolynomialFeatures(degree=3,include_bias=True)

    x_t_poly = poly.fit_transform(x_combo)

    reg = LinearRegression(fit_intercept=False)

    reg.fit(x_t_poly,u_combo)

    # ##################

    X_poly = poly.fit_transform(truth_set_shorter[:-1,:].reshape(-1,1))
    U_d = reg.predict(X_poly).reshape(-1,k)
    h_t_true = U_true_shorter - U_d

    h_time_plus_one = h_t_true[1:]
    h_time = h_t_true[:-1]

    residual = phi*h_time - h_time_plus_one
    
    poly_acf_residual = np.ones((8,1000))

    for i in range(8):
            poly_acf = acf(residual[:1000000,i],adjusted=True,nlags=nlags-1)
            poly_acf_residual[i,:] = poly_acf
    
    np.save("acf_data/poly_acf_residual.npy",poly_acf_residual)




    #### rnn #####

    u_det_model = keras.models.load_model("../../saved_models/rnn/u_nn.h5")
    rnn_model = keras.models.load_model("../../saved_models/rnn/rnn_update.h5")
    x_mean=3.737599263
    x_std=5.08148659615
    u_mean=3.79456864136
    u_std=4.65478026

    # ##### 20 ######

    rnn_truth_set_20 = np.concatenate([truth_set_shorter[:,:,np.newaxis],20*np.ones((truth_set_shorter[:].shape[0],k,1))],axis=2)

    def create_dataset_with_u(climate_eval_dataset,dt=0.005):
        """Shape of climate_eval_dataset is num_steps x k x 2"""
        
        data = climate_eval_dataset[:,:,0] #so we get num_steps x k shape    
        u_data =  U(data[:-1,:],data[1:,:],climate_eval_dataset[0,0,-1])    
        x_prev = data[:-1,:]
        
        return np.stack([u_data,x_prev],axis=2)

    def prep_holdout(test_dataset):
        return np.transpose(test_dataset,[1,0,2])

    def scaled_data(input_dataset,x_mean=3.737599263,x_std=5.08148659615,u_mean=3.79456864136,u_std=4.65478026):
        data = input_dataset.copy()
        data[:,:,0] = (data[:,:,0] - u_mean)/u_std 
        data[:,:,1] = (data[:,:,1] - x_mean)/x_std 
        return data


    test_dataset_20_new = create_dataset_with_u(rnn_truth_set_20)   
    test_nn_features_20 = prep_holdout(test_dataset_20_new)
    test_nn_features_20 = scaled_data(test_nn_features_20)
    test_nn_20_input = test_nn_features_20[:,:-1,:]
    test_nn_20_output = test_nn_features_20[:,1:,:]
    hidden_in_test_20 = np.zeros(shape=(test_nn_features_20.shape[0],4))

    u_det = u_det_model.predict(test_nn_20_input[:,:,-1:])
    u_true = test_nn_20_input[:,:,:1]
    r = u_true - (u_det-u_mean)/u_std

    x_mean_out = rnn_model.predict([r,hidden_in_test_20,hidden_in_test_20])[0]

    u_det_target = u_det_model.predict(test_nn_20_output[:,:,-1:])
    u_true_target = test_nn_20_output[:,:,:1]
    r_target = u_true_target - (u_det_target-u_mean)/u_std

    residual = r_target - x_mean_out
    residual = residual[:,:,0].transpose([1,0])

    rnn_acf_residual = np.ones((8,1000))

    for i in range(8):
        rnn_acf = acf(residual[:1000000,i],adjusted=True,nlags=nlags-1)
        rnn_acf_residual[i,:] = rnn_acf

    np.save("acf_data/rnn_acf_residual.npy",rnn_acf_residual)

