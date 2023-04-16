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
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import time

    

def simulate_polynomial_param_weather(starting_condition,num_steps,num_ensemble_members=40,
                                    sigma = 2.01156,phi =0.985728 ,F=20,dt=0.005):
    
    
    def _rhs_X_dt(X, U):
        """Compute the right hand side of the X-ODE."""

        dXdt = (-np.roll(X, 1,axis=1) * (np.roll(X, 2,axis=1) - np.roll(X, -1,axis=1)) -
                    X + F - U)
            
        return dt * dXdt 
    
    def sub_grid_u_term(X,array_h):
        h_t_minus1 = array_h
        X_poly = poly.fit_transform(X.reshape(-1,1))
        U_d = reg.predict(X_poly).reshape(-1,k)
        z = np.random.normal(size=(length,k))
        U_pred = U_d + h_t_minus1
        h_t = phi*h_t_minus1 + sigma*((1-phi**2)**0.5)*z
        return U_pred,h_t
    
    
    def step(X_in,array_h):
        """Integrate one time step"""
        X = X_in.copy()
        k1_X = _rhs_X_dt(X, U=0)
        k2_X = _rhs_X_dt(X + k1_X / 2, U=0)

        U,h_t = sub_grid_u_term(X,array_h) ######
        
        X += k2_X
        
        X += - U * dt
        return X,h_t
        
    initX = starting_condition.repeat(num_ensemble_members,axis=0)  
    
    length = initX.shape[0]
    
    array = np.zeros(shape=(length,k,num_steps))
    array[:,:,0] = initX
    
    array_h = sigma*np.random.normal(size=(length,k)) 
    
    for i in range(num_steps-1):
        print(i)
        X_new, h_new = step(array[:,:,i],array_h)
        array[:,:,i+1] = X_new
        array_h = h_new
        
    return array.reshape(starting_condition.shape[0],num_ensemble_members,k,num_steps)

        

if __name__ == "__main__":
    
    ######### USAGE ###########
    
    parser = argparse.ArgumentParser()
   
    parser.add_argument("F", type=int, help="F")
    parser.add_argument("path_to_init_data", type=str, help="path_to_init_data")

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


    poly = PolynomialFeatures(degree=3,include_bias=True)

    x_t_poly = poly.fit_transform(x_combo)

    reg = LinearRegression(fit_intercept=False)

    reg.fit(x_t_poly,u_combo)



##################################

    truth_weather = np.load(args.path_to_init_data)


    truth_weather_init_conditions = truth_weather[:,0,:]
    num_init_conds = truth_weather.shape[0]
    length_simulations = truth_weather.shape[1]
    num_ensemble_members = 40


    
    array = simulate_polynomial_param_weather(truth_weather_init_conditions,length_simulations)    
    np.save("../../data/simulation_runs/polynomial_param/weather_array_{}.npy".format(F),array)
