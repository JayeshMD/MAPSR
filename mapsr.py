#!/usr/bin/env python
# coding: utf-8

import os
import sys
import gc
import pickle
import torch
import torch.optim as optim
import numpy as np

# From mapsr
import ffnn
import mapsr_utils
import schemes_dev as sc

def print_there(x, y, text):
     sys.stdout.write("\x1b7\x1b[%d;%df%s\x1b8" % (x, y, text))
     sys.stdout.flush()

dtype = torch.float32
cpu   = torch.device('cpu')

#============================================== 
# Model adaptive phase space reconstruction
#==============================================

def mapsr(args, comm):
    my_rank = comm.Get_rank()

    ffnn.write_nn(args.folder, args.n_layers, args.n_nodes)
    
    sys.path.append(args.folder)
    from neuralODE import ODEFunc
        
    # # Define device
    device = mapsr_utils.set_device(args)
    args.to(device)

    if args.adjoint:
        from torchdiffeq import odeint_adjoint as odeint
    else:
        from torchdiffeq import odeint

    # # Folder to save output
    try:
        os.makedirs(args.folder)
    except:
        print("Folder", args.folder, "already exist.")

    # Obtain Time Series
    t_true, x_true, dt = mapsr_utils.load_data(args, device, dtype)
    τ_arr, dim         = mapsr_utils.initialize_delay_vector(x_true, args, device, dt)
    
    # If $|\tau_i-\tau_j|<tau_th$ then $\tau_i$ and $\tau_j$ will merge
    tau_th = 0.5*dt
    
    # learning rate
    lr   = lambda iteration: mapsr_utils.get_learning_rate(iteration, args.lr_start    , args.lr_end    , args.niters/2, 1000)
    lr_τ = lambda iteration: mapsr_utils.get_learning_rate(iteration, args.lr_tau_start, args.lr_tau_end, args.niters/2, 1000) 

    # Initialize function and optimizer
    func = ODEFunc(dimensions=dim).to(device)
    optimizer = optim.RMSprop(func.parameters(), lr=lr(0))
    optimizer_τ = optim.RMSprop(τ_arr, lr=lr_τ(0))

    # Variables to be monitored
    iterations = []
    loss_arr_save = []
    τ_arr_save = []

    if args.restart:
        func,τ = mapsr_utils.restart(args, device, func, τ)  

    for kk in range(args.niters):
     
        iterations.append(kk)
        
        batch_time = args.batch_time #+ (kk//1000)*10
        func, optimizer, optimizer_τ, τ_arr = mapsr_utils.get_fun(args, func, optimizer, optimizer_τ, lr(kk), lr_τ(kk), τ_arr, tau_th)
        
        optimizer.zero_grad()
        optimizer_τ.zero_grad()

        t_batch, z_batch = mapsr_utils.get_batch(t_true, x_true, args.Nc, τ_arr, batch_time, args.batch_size, device=device)        
        z_pred = odeint(func, z_batch[0,:,:].reshape(z_batch.shape[1], z_batch.shape[2]), t_batch[:,0], options={'dtype':dtype}).to(device)
        
        #loss = torch.mean(torch.abs(z_pred - z_batch)) + 1e-10*sum_weights(func,power=2)+ 1e-6*torch.sum(torch.abs(τ/dt))
        loss = torch.mean(torch.sum(torch.abs(z_pred-z_batch),axis=2))
        #loss = torch.mean(torch.sum(torch.abs(z_pred-z_batch),axis=2)) + 1e-9 * get_sum(τ_arr)
        #loss = torch.mean(torch.sum(torch.abs(z_pred-z_batch)**2,axis=2))
        #loss = sum_weights(func,power=1)  # To test if weights are becoming zero => Test ok
        #loss = get_sum(τ_arr)
        loss.backward()
        
        loss_arr_save.append(loss.to(cpu).detach().numpy())
        optimizer.step()
        
        if kk>10:
            optimizer_τ.step()
            with torch.no_grad():
                τ_min = 1
                for i in range(len(τ_arr)):
                    if τ_arr[i][0]<τ_min:
                        τ_min = τ_arr[i][0]
                for i in range(len(τ_arr)):
                    τ_arr[i] = τ_arr[i] - τ_min

        τ_arr_temp = []
        for i in range(len(τ_arr)):
            τ_arr_temp.append(τ_arr[i].to(cpu).detach().numpy())

        τ_arr_save.append(τ_arr_temp)

        if kk%args.test_freq==0:  

            pred_file = args.folder +'/comp_pred_'+str(kk)+'.pkl'

            with open(pred_file,'wb') as pred_file_open:
                pickle.dump(['Iter:'+ str(kk), kk, t_batch, z_batch, t_batch, z_pred, τ_arr], pred_file_open)

            with open(args.folder+'/delay.pkl', 'wb') as fl:
                pickle.dump(τ_arr_save, fl)

            with open(args.folder+'/loss.pkl', 'wb') as fl:
                pickle.dump(loss_arr_save, fl)

            torch.save(func.state_dict(), args.folder + '/Weight.pkl')