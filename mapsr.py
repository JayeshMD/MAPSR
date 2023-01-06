#!/usr/bin/env python
# coding: utf-8

# In[1]:

import os
import time
import sys
import platform
import gc

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

import schemes_dev as sc
import ffnn

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": "Helvetica",
})

#============================================== 
# Model adaptive phase space reconstruction
#==============================================
def mapsr(args):
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
        # response = input("Warning: Folder alrady exist. Figures will be replaced. Do you want to replace? (y/n)")
        # if response.lower()!='y':
        #     args.savefig= False


    # # Define function to plot time series

    def plot_timeseries(t,x):
        if len(x.shape)==1:
            x = x.reshape(-1,1)
        
        n = x.shape[1]
        
        fig = plt.figure()
        axs = [fig.add_subplot(n, 1, i+1) for i in range(n)]
        for i in range(n):
            axs[i].plot(t, x[:,i], label = '$x_'+str(i+1)+'$')
            axs[i].set_xlim([min(t), max(t)])
            axs[i].set_xlabel('$t$', fontsize=20)
            axs[i].set_ylabel('$x_'+str(i+1)+'$',fontsize=20)
            #axs[i].legend(loc='upper right')
        plt.tight_layout()
        fig.clf()
        plt.close(fig)

        # if args.viz: 
        #     plt.show()


    # # Obtain Time Series

    data = np.loadtxt(args.datafile)

    t_true = torch.tensor(data[:,0])
    x_data = torch.tensor(data[:,1:])

    plot_timeseries(t_true, x_data)

    # # Normalized time series
    x_data_sam = x_data
    x_data_sam = x_data_sam- x_data_sam.mean(0)
    x_data_sam = x_data_sam/x_data_sam.abs().max(0).values

    plot_timeseries(t_true, x_data_sam)

    x_true = (x_data_sam[:,0]).type(dtype)
    t_true = (t_true-t_true[0]).type(dtype)


    # # Time step

    dt = t_true[1] - t_true[0]
    x_true = x_true.to(device)
    dt = dt.to(device)
    dt


    # # Method to creates batch_size number of batches of true data of batch_time duration

    def get_batch(t, x, Nc, τ, batch_time, batch_size, device= torch.device('cpu')):
        dt = (t[1]-t[0]).to(device)
        assert τ.max()<Nc*dt, "Maximum value of delay should be less than Nc*dt="+str(Nc*dt)+'.'
        
        #print('main dt:', dt.device)
        t = t.to(device)
        
        z_true = sc.interp_linear(t, x, Nc, τ, device=device)
        id_sel = torch.randint(0, z_true.shape[0] - batch_time-1, (batch_size,))
        z_true_stack = torch.stack([z_true[id_sel + i, :] for i in range(batch_time)], dim=0)
        t_true_stack = torch.stack([t_true[id_sel + i] for i in range(batch_time)], dim=0)
        return t_true_stack.to(device), z_true_stack.to(device)


    # # Integrate the <i>fun</i> from initial state $z_0$ for given time array $t$

    def get_pred(fun, z0, t):
        dt = len(t)
        z_pred = z0
        for i in range(1, len(t)):
            z_next = args.method(fun, t[i], z_pred[i - 1, :, :], dt)
            z_pred = torch.cat([z_pred, z_next.reshape(1, z_pred.shape[1], z_pred.shape[2])], 0)
        return z_pred.to(device)

    # # Function to modify delay vector and return modified ODE function 
    def get_fun(func, optimizer, optimizer_τ, lr, lr_τ, τ, acc):
        l_0 = len(τ)
        τ = sc.merge(τ, acc)
        
        if l_0>len(τ):
            print('Merged τ:',τ)
            func = ODEFunc(dimensions=len(τ))
            
        τ = τ.clone().detach().requires_grad_(True)
        optimizer = optim.RMSprop(func.parameters(), lr=lr)   # lr=1e-4
        optimizer_τ = optim.RMSprop([τ], lr=lr_τ)             # lr=1e-6 => Gives ok results
        
        return func, optimizer, optimizer_τ, τ


    # # Function to compute sum of parameters generally for regularization
    def sum_weights(func, power= 2):
        i = 0
        for params in func.parameters():
            if i==0:
                s = torch.sum(params**power)
                i+=1
            else:
                s+= torch.sum(params**power)
        return s


    # # Function to plot 2D plots
    def plot_cmp(fig, title, kk, tt,zt, tp,zp, τ):
        cpu = torch.device('cpu')

        right = 0.6
        top   = 0.8
        
        fig.clf()
        fig.suptitle(title, fontsize=25)
        if zt.shape[2]==1:
            axs_arr = fig.add_subplot(1,1,1)
            for p_id in range(args.batch_size):
                axs_arr.plot(zt[:, p_id, 0].to(cpu).detach().numpy(),'k-')
                axs_arr.plot(zp[:, p_id, 0].to(cpu).detach().numpy(),'r--', linewidth=2)
                axs_arr.set_ylabel('$x(t)$',fontsize=20)
            axs_arr.set_xlabel('$t_{id}$',fontsize=20)
            text = '$\\tau_'+str(0)+'='+ "{:.4f}".format(τ[0]) +'$'
            axs_arr.set_title(text,fontsize=20)
            # axs_arr.text(right, top, text, fontsize=20,
            #             horizontalalignment='center',
            #             verticalalignment='center',
            #             #rotation='vertical',
            #             transform=axs_arr.transAxes)
        else:
            axs_arr = [fig.add_subplot(zt.shape[2]-1,1,i+1) for i in range(zt.shape[2]-1)] 
            axs_arr[0].set_title(title, fontsize=25)
            for i in range(zt.shape[2]-1):
                text = '$\\tau_'+str(i+1)+'='+ "{:.4f}".format(τ[i+1]) +'$'
                axs_arr[i].set_title(text, fontsize=20)
                # axs_arr[i].text(right, top, text, fontsize=20,
                #             horizontalalignment='center',
                #             verticalalignment='center',
                #             #rotation='vertical',
                #             transform=axs_arr[i].transAxes)

                for p_id in range(args.batch_size):
                    axs_arr[i].plot(zt[:, p_id, 0].to(cpu).detach().numpy(), zt[:, p_id, i+1].to(cpu).detach().numpy(),'k-')
                    axs_arr[i].plot(zp[:, p_id, 0].to(cpu).detach().numpy(), zp[:, p_id, i+1].to(cpu).detach().numpy(),'r--', linewidth=2)    
                    axs_arr[i].set_ylabel('$x(t+\\tau_'+str(i+1)+'$)',fontsize=20)
            
            axs_arr[i].set_xlabel('x(t)',fontsize=20)
        
        fig.tight_layout()
        fig.subplots_adjust(wspace=0.4,
                            hspace=0.4) 

        
            
        
        if args.savefig:
            #fig.canvas.draw()
            fig_name = args.folder+'/'+str(kk)+'.png'
            #print('Saving figure:', fig_name)
            fig.savefig(fig_name)
        
        # if args.viz:   
        #     plt.show(block=False)
        
        plt.pause(1e-3)
        fig.clf()
        plt.close(fig)

    def plot_loss(fig, title, it, loss):
        cpu = torch.device('cpu')
        
        fig.clf()
        axs = fig.add_subplot(1,1,1)
        axs.plot(it, loss,'k-')
        axs.set_title(title, fontsize=25)
        axs.set_xlabel('Iteration',fontsize=20)
        axs.set_ylabel('$\mathcal{L}$',fontsize=20)
        plt.tight_layout()

        if args.savefig:
            #fig.canvas.draw()
            fig_name = args.folder+'/loss.png'
            fig.savefig(fig_name)
            #print('Saved figure:', fig_name)
        
        # if args.viz:    
        #     plt.show(block=False)
        
        plt.pause(1e-3)
        fig.clf()
        plt.close(fig)


    def plot_delays(fig, title, it, τ_arr):
        cpu = torch.device('cpu')
        
        fig.clf()
        axs = fig.add_subplot(1,1,1)
        m = len(it)
        for i in range(m):
            axs.plot(it[i]*np.ones(len(τ_arr[i])), τ_arr[i],'ko')
        axs.set_title(title, fontsize=25)
        axs.set_xlabel('Iteration',fontsize=20)
        axs.set_ylabel('$\\tau$',fontsize=20)
        plt.tight_layout() 
        
        if args.savefig:
            #fig.canvas.draw()
            fig_name = args.folder+'/delay_evol.png'
            fig.savefig(fig_name)
            #print('Saved figure:', fig_name)

        # if args.viz:  
        #     plt.show(block=False)
        
        plt.pause(1e-3)
        fig.clf()
        plt.close(fig)

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


    # # Initialize delay vector $\tau$ and $acc$. If $|\tau_i-\tau_j|<acc$ then $\tau_i$ and $\tau_j$ will merge

    τ = torch.linspace(0,args.Nc*dt, args.dimensions+1)
    τ = τ[0:args.dimensions].to(device).clone().detach().requires_grad_(True)
    acc = 0.5*dt

    # # Initialize function and optimizer

    func = ODEFunc(dimensions=len(τ)).to(device)
    optimizer = optim.RMSprop(func.parameters(), lr=args.lr)
    optimizer_τ = optim.RMSprop([τ], lr=args.lr_τ)


    # # Variables to be monitored

    iterations = []
    loss_arr = []
    τ_arr = []


    # # Main function

    
    if args.restart:
        func,τ = restart(func,τ)

    for kk in range(args.niters):
        iterations.append(kk)
        
        batch_time = args.batch_time #+ (kk//1000)*10
        func, optimizer, optimizer_τ, τ = get_fun(func, optimizer, optimizer_τ, args.lr, args.lr_τ, τ, acc)
        
        optimizer.zero_grad()
        optimizer_τ.zero_grad()

        t_batch, z_batch = get_batch(t_true, x_true, args.Nc, τ, batch_time, args.batch_size, device=device)
        
        #print(z_batch.shape)
        
        z_pred = odeint(func, z_batch[0,:,:].reshape(z_batch.shape[1], z_batch.shape[2]), t_batch[:,0], options={'dtype':dtype}).to(device)
        
        #loss = torch.mean(torch.abs(z_pred - z_batch)) + 1e-10*sum_weights(func,power=2)+ 1e-6*torch.sum(torch.abs(τ/dt))
        loss = torch.mean(torch.sum(torch.abs(z_pred-z_batch),axis=2))
        loss.backward()
        loss_arr.append(loss.to(cpu).detach().numpy())
        #print('loss:',loss)
        optimizer.step()
        
        if kk>10:
            optimizer_τ.step()
            with torch.no_grad():
                τ[0] = 0.0
        
        τ_arr.append(τ.to(cpu).detach().numpy())
            
        if kk%args.test_freq==0:
            print('Iter:',kk)
            #print('τ:',τ)
            #print('w0:',func.parameters())
            #plot_cmp(fig_cmp, 'Iter:'+ str(kk)+', '+'$\\tau$='+str(τ.to(cpu).detach().numpy()), kk, t_batch, z_batch, t_batch, z_pred)
            fig_cmp = plt.figure(figsize=[10,15])

            fig_loss = plt.figure(figsize=[10,6])
            title_loss = "Iterations vs Loss"

            fig_delay = plt.figure(figsize=[10,6])
            title_delay = "Iterations vs Delay"

            plot_cmp(fig_cmp, 'Iter:'+ str(kk), kk, t_batch, z_batch, t_batch, z_pred, τ.to(cpu).detach().numpy())
            plot_loss(fig_loss, title_loss, iterations, loss_arr)
            plot_delays(fig_delay, title_delay, iterations, τ_arr)
            np.savetxt(args.folder+'/delay.txt',τ_arr)
            np.savetxt(args.folder+'/loss.txt',loss_arr)
            torch.save(func.state_dict(), args.folder + '/Weight.pkl')

            plt.close("all")
            plt.close()
            gc.collect()