#!/usr/bin/env python
# coding: utf-8

# In[1]:

import os
import time
import sys
import platform
import gc

import numpy as np
import pickle

import torch
import torch.nn as nn
import torch.optim as optim

import schemes_dev as sc
import ffnn

def print_there(x, y, text):
     sys.stdout.write("\x1b7\x1b[%d;%df%s\x1b8" % (x, y, text))
     sys.stdout.flush()

#from memory_profiler import profile

#============================================== 
# Model adaptive phase space reconstruction
#==============================================
#@profile
def mapsr(args, comm):
    my_rank = comm.Get_rank()
    #args = model_param()
    
    ffnn.write_nn(args.__dict__['folder'],\
                  args.__dict__['n_layers'],\
                  args.__dict__['n_nodes'],)

    
    sys.path.append(args.__dict__['folder'])
    from neuralODE import ODEFunc
        
    cpu = torch.device('cpu')
    dtype = torch.float32

    # # Define device

    if args.use_gpu:
        if platform.system()=="Darwin":
            try:
                device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
            except:
                device = torch.device("cpu")        
        else:
            device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")
        #device = torch.device('cpu')   # Remove once code is compatible with gpu 
    else:
        device = cpu

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

    # # Obtain Time Series

    data = np.loadtxt(args.datafile)

    t_true = torch.tensor(data[args.start_id:args.end_id,0])
    x_data = torch.tensor(data[args.start_id:args.end_id,1:])
    

    # # Normalized time series
    x_data_sam = x_data
    x_data_sam = x_data_sam- x_data_sam.mean(0)
    x_data_sam = x_data_sam/x_data_sam.abs().max(0).values

    x_true_sam = (x_data_sam[:,0::]).type(dtype).to(device)
    t_true_sam = (t_true-t_true[0]).type(dtype).to(device)

    
    # # Time step
    t_true = t_true_sam
    dt = t_true[1] - t_true[0]
    
    x_true = [ x_true_sam[:,0], x_true_sam[:,1] ]
    dt = dt.to(device)
    dt


    # # Method to creates batch_size number of batches of true data of batch_time duration

    def get_batch(t, x, Nc, τ_arr, batch_time, batch_size, device= torch.device('cpu')):
        dt = (t[1]-t[0]).to(device)
        for τ in τ_arr:
            assert τ.max()<Nc*dt, "Maximum value of delay should be less than Nc*dt="+str(Nc*dt)+'.'
        
        #print('main dt:', dt.device)
        t = t.to(device)
        
        z_true = sc.interp_linear_multi([t,t], x, Nc, τ_arr, device=device)
        id_sel = torch.randint(0, z_true.shape[0] - batch_time-1, (batch_size,))
        z_true_stack = torch.stack([z_true[id_sel + i, :] for i in range(batch_time)], dim=0)
        t_true_stack = torch.stack([t_true[id_sel + i] for i in range(batch_time)], dim=0)
        return t_true_stack.to(device), z_true_stack.to(device)

    # # Function to modify delay vector and return modified ODE function 
    def get_fun(func, optimizer, optimizer_τ, lr, lr_τ, τ_arr, acc):

        τ_all_1, W_all_1, b_all_1 = sc.merge_multi(τ_arr, func, acc)

        dim = 0
        for τ in τ_all_1:
            dim += len(τ)

        func = ODEFunc(dimensions=dim) 
        sc.set_all_weight_mat(func, W_all_1)
        sc.set_all_bias_mat(func, b_all_1)
            
        optimizer = optim.RMSprop(func.parameters(), lr=lr)   # lr=1e-4
        optimizer_τ = optim.RMSprop(τ_all_1, lr=lr_τ)             # lr=1e-6 => Gives ok results
        
        return func, optimizer, optimizer_τ, τ_all_1


    # # Function to compute sum of parameters generally for regularization
    def sum_weights(func, power= 2):
        i = 0
        for params in func.parameters():
            if i==0:
                s = torch.sum(torch.abs(params)**power)
                i+=1
            else:
                s+= torch.sum(torch.abs(params)**power)
        return s

    def restart(fun,τ):
        print("\n"*3, "Loading state....")
        func.load_state_dict(torch.load(args.folder + '/Weight.pkl'))
        delay_temp = np.loadtxt(args.folder+'/delay.txt')
        if len(delay_temp.shape)==1:
            τ = torch.tensor([delay_temp[-1]]).to(device).clone().detach().requires_grad_(True)
        else:
            τ = torch.tensor(delay_temp[-1,:]).to(device).clone().detach().requires_grad_(True)
        print("Initial τ:",τ)
        print("Loading successful!"+"\n"*3)
        return fun,τ

    def get_sum(arr, power=1):
        s = torch.sum(torch.abs(arr[0])**power)
        for i in range(1,len(arr)):
            s = s + torch.sum(torch.abs(arr[i])**power)
        return s
    
    def get_learning_rate(iteration, lr_start, lr_end, loc_trans, spread):
        lr =  lr_start*0.5*(1-np.tanh((iteration-loc_trans)/spread)) \
            + lr_end  *0.5*(1+np.tanh((iteration-loc_trans)/spread))
        return lr

    # # Initialize delay vector $\tau$ and $acc$. If $|\tau_i-\tau_j|<acc$ then $\tau_i$ and $\tau_j$ will merge
    τ_arr = []
    for i in range(len(x_true)):
        #τ = torch.linspace(0,args.Nc*dt, args.dimensions+1)
        τ = torch.linspace(0,(args.dimensions-1) *args.dtau * dt, args.dimensions)
        τ_arr.append(τ[0:args.dimensions].to(device).clone().detach().requires_grad_(True))
    acc = 0.5*dt

    dim = 0
    for τ in τ_arr:
        dim += len(τ)


    # # learning rate
    lr   = lambda iteration: get_learning_rate(iteration, args.lr_start    , args.lr_end    , args.niters/2, 1000)
    lr_τ = lambda iteration: get_learning_rate(iteration, args.lr_tau_start, args.lr_tau_end, args.niters/2, 1000) 

    # # Initialize function and optimizer
    
    func = ODEFunc(dimensions=dim).to(device)
    optimizer = optim.RMSprop(func.parameters(), lr=lr(0))
    optimizer_τ = optim.RMSprop([τ], lr=lr_τ(0))


    # # Variables to be monitored

    iterations = []
    loss_arr_save = []
    τ_arr_save = []


    # # Main function

    
    if args.restart:
        func,τ = restart(func,τ)
    
    #comm.Barrier()
    os.system('clear')
    #comm.Barrier()

    # lr_1 = []
    # lr_2 = []

    # for kk in range(args.niters):
    #     #print("kk:", kk)
    #     lr_1.append(lr(kk))
    #     lr_2.append(lr_τ(kk))

    # import matplotlib.pyplot as plt
    # fig = plt.figure()
    # axs = fig.add_subplot(1,1,1)
    # axs.plot(lr_1)
    # axs.plot(lr_2)
    # fig.savefig('lr_test.png')

    # exit()

    for kk in range(args.niters):
        #print("kk:", kk)
     
        iterations.append(kk)
        
        batch_time = args.batch_time #+ (kk//1000)*10
        func, optimizer, optimizer_τ, τ_arr = get_fun(func, optimizer, optimizer_τ, lr(kk), lr_τ(kk), τ_arr, acc)
        
        optimizer.zero_grad()
        optimizer_τ.zero_grad()

        t_batch, z_batch = get_batch(t_true, x_true, args.Nc, τ_arr, batch_time, args.batch_size, device=device)
        
        #print(z_batch.shape)
        
        z_pred = odeint(func, z_batch[0,:,:].reshape(z_batch.shape[1], z_batch.shape[2]), t_batch[:,0], options={'dtype':dtype}).to(device)
        
        #loss = torch.mean(torch.abs(z_pred - z_batch)) + 1e-10*sum_weights(func,power=2)+ 1e-6*torch.sum(torch.abs(τ/dt))
        #loss = torch.mean(torch.sum(torch.abs(z_pred-z_batch),axis=2))
        loss = torch.mean(torch.sum(torch.abs(z_pred-z_batch),axis=2)) #+ 1e-9 * get_sum(τ_arr)
        #loss = torch.mean(torch.sum(torch.abs(z_pred-z_batch)**2,axis=2))
        #loss = sum_weights(func,power=1)  # To test if weights are becoming zero => Test ok
        #loss = get_sum(τ_arr)
        loss.backward()
        
        loss_arr_save.append(loss.to(cpu).detach().numpy())
        #print('loss:',loss)
        optimizer.step()
        
        if kk>10:
            optimizer_τ.step()
            with torch.no_grad():
                #min_id = 0
                τ_min = 1
                for i in range(len(τ_arr)):
                    if τ_arr[i][0]<τ_min:
                        τ_min = τ_arr[i][0]
                        #min_id = i
                for i in range(len(τ_arr)):
                    τ_arr[i] = τ_arr[i] - τ_min

        τ_arr_temp = []
        for i in range(len(τ_arr)):
            τ_arr_temp.append(τ_arr[i].to(cpu).detach().numpy())
        #print("len(τ_arr[1]):", len(τ_arr[1]))


        #print("τ_arr_temp:", τ_arr_temp)
        τ_arr_save.append(τ_arr_temp)

        if kk%args.test_freq==0:  
            sys.stdout.write("\x1b7\x1b[%d;%df%s\x1b8" % (my_rank+1, 0,'CPU-'+str(my_rank)+'-> Iter:'+str(kk)))
            sys.stdout.flush()

            pred_file = args.folder +'/comp_pred_'+str(kk)+'.pkl'

            with open(pred_file,'wb') as pred_file_open:
                pickle.dump(['Iter:'+ str(kk), kk, t_batch, z_batch, t_batch, z_pred, τ.to(cpu).detach().numpy()], pred_file_open)

            with open(args.folder+'/delay.pkl', 'wb') as fl:
                pickle.dump(τ_arr_save, fl)

            with open(args.folder+'/loss.pkl', 'wb') as fl:
                pickle.dump(loss_arr_save, fl)

            torch.save(func.state_dict(), args.folder + '/Weight.pkl')
        gc.collect()
# %%
