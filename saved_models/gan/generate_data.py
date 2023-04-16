#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import sklearn
import time
from pickle import load


from sklearn.preprocessing import StandardScaler,PowerTransformer
import math
from scipy.stats import multivariate_normal
from scipy.stats import norm
import os
import pickle
import argparse
import tensorflow as tf
from tensorflow import keras
K = keras.backend
from helper import *
import time


def simulate_gan_param(initX,num_steps, sigma =0.705908 ,phi =0.70830,F=20,dt=0.005):
    
    """
    lorenz.py file in gagne repository has def run_lorenz_96_forecast which uses config from forecast_gan_travis.yaml

    Random_updater is what takes in z_(t-1) and applies AR1 to give z_t, which is used to create u. 
    
    num steps is #MTU/0.005
    """
    
    
    def _rhs_X_dt(X, U):
        """Compute the right hand side of the X-ODE."""

        dXdt = (-np.roll(X, 1) * (np.roll(X, 2) - np.roll(X, -1)) -
                    X + F - U)
            
        return dt * dXdt 
    
    def sub_grid_u_term(X,array_h):
        h_t_minus1 = array_h
        #scale X
        X_scaled = scaler1.transform(X.reshape(k,1))
        #generate noise and u term 
        noise = np.random.normal(size=(k,codings_size))
        h_t = phi*h_t_minus1 + noise*sigma #red noise (eqs 7b and 7c from gagne paper)
        U_scaled = generator([X_scaled,h_t])
        #unscale u
        U = scaler2.inverse_transform(U_scaled)
        return U.reshape(k),h_t
    
    
    def step(X_in,array_h):
        """Integrate one time step"""
        X = X_in.copy()
        k1_X = _rhs_X_dt(X, U=0)
        k2_X = _rhs_X_dt(X + k1_X / 2, U=0)

        U,h_t = sub_grid_u_term(X,array_h) ######
        
        X += k2_X
        
        X += - U * dt
        return X,h_t
        
        
    #np.random.seed(42)
    
    array = np.zeros(shape=(k,num_steps))
    array[:,0] = initX
    
    array_h = np.random.normal(size=(k,codings_size))
    
    for i in range(num_steps-1):
        X_new, h_new = step(array[:,i],array_h)
        array[:,i+1] = X_new
        array_h = h_new
        if i % 200000 == 0:
            np.save("../../data/simulation_runs/gan_gagne/corrected_gans/new_generated_data_{}.npy".format(F),array)
        
    return array

def u_deriver(x_train,F,save_time_step):
    u_t = -np.roll(x_train[:-1,:],1,axis=1) * (np.roll(x_train[:-1,:],2,axis=1) - np.roll(x_train[:-1,:],-1,axis=1)) - x_train[:-1,:] + F -(x_train[1:,:] - x_train[:-1,:])/save_time_step
    return u_t

if __name__ == "__main__":
    
    ######### USAGE ###########
    
    parser = argparse.ArgumentParser()
   
    parser.add_argument("F", type=int, help="F")
    parser.add_argument("path_to_init_data", type=str, help="path_to_init_data")
    parser.add_argument("mtu", type=int, help="mtu")

    args = parser.parse_args()



if __name__ == "__main__":
    
    ######### USAGE ###########

    
    k = 8
    J = 32
    save_time_step = 0.005
    h=1 
    c=10
    b=10
   
    F = args.F
    mtu = args.mtu

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



##################################
########### Load model ##############
##################################

    generator = keras.models.load_model("gan_generator_final.h5")

    codings_size = 1 + 1 + 2*16


##################################
########### Generate data ##############
##################################


    x_climate = np.load(args.path_to_init_data)
    init_X_climate = x_climate[0,:,0]

    num_steps_climate = int(mtu/0.005)

    start = time.time()

    data = simulate_gan_param(init_X_climate,num_steps_climate,F=F)
    np.save("../../data/simulation_runs/gan_gagne/corrected_gans/new_generated_data_{}.npy".format(F),data)

    end = time.time()
    print(end-start)