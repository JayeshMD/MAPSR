import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import pickle as pkl

import matplotlib.pyplot as plt

class ami_fnn_analysis:
    def __init__(self, param_file, var_names, case_names):
        self.param_file = param_file
        self.case_names = case_names
        self.param_train = pd.read_csv(param_file)
        self.var_names = var_names
        self.sampling_time = self.set_sampling_time()
        self.var_idx = self.set_variable()
        self.time_series = dict()
        self.t = dict()
        self.ami = dict()
        self.tau_ami = dict()
        self.dim = dict()
        self.fnn = dict()
        self.fnn_zero = dict()
        self.dim_sel = dict()
        self.delay_vec_sel = dict()


    def set_sampling_time(self):
        f = self.param_train.iloc[0].datafile
        data = np.loadtxt(f)            
        t = data[:,0]
        sampling_time = t[1]-t[0]
        
        resp = input("Current sampling time is "+ str(sampling_time)+". Do you want to reset it?(y/n)")
        if resp.lower() == 'y':
            sampling_time = float(input("New sampling time:"))
        return sampling_time
        


    def set_variable(self):
        param_train = self.param_train
        n_var = len(param_train.iloc[0].timeseries_index)
        if n_var>1:
            print("MAPSR analysis is performed with time series:", param_train.iloc[0].timeseries_index)
            print("Name of time series:", self.var_names)
            idx = int(input("Time series index for computing AMI:"))
        else:
            idx = param_train.iloc[0].timeseries_index[0]
        print("The AMI-FNN will be performed for variable:", self.var_names[idx])

        return idx

    def get_ami_fnn(self):
        param_train = self.param_train
        sampling_time = self.sampling_time

        for i in range(len(param_train)):
            f = param_train.iloc[i].datafile
            data = np.loadtxt(f)
            
            #print("data.shape",data.shape)

            data_sel = data[param_train.iloc[i].start_id:param_train.iloc[i].end_id,:]

            x = data_sel[:,1:]
            #print('x.shape:',x.shape)
            self.time_series[f] = x[:,self.var_idx]
            self.t[f] = np.arange(len(self.time_series[f]))* self.sampling_time
            
            fnn_threshold = len(self.time_series[f])*0.1/100

            #print("self.time_series[f].shape",self.time_series[f].shape)

            self.ami[f], self.tau_ami[f], self.dim[f], self.fnn[f], self.fnn_zero[f], self.dim_sel[f], self.delay_vec_sel[f] = ami_fnn(x = self.time_series[f], 
                                                                                                                      sampling_time = self.sampling_time, 
                                                                                                                      tau_max = 500, 
                                                                                                                      fnn_threshold = fnn_threshold, 
                                                                                                                      win =2, 
                                                                                                                      dim_max = 15)

    def create_table(self, case_name, norm_fact):
        nf_str = "$\\times10^{"+str(int(np.log10(norm_fact)))+"}$"

        tab = pd.DataFrame(columns = ["Case", "Method", "Dimension", "Delay ("+nf_str+")"])
        tab_sup = pd.DataFrame(columns = ["Case", "Method", "Dimension", "Delay ("+nf_str+")"])

        for i in range(len(self.param_train)):
            f = self.param_train.iloc[i].datafile

            formated_delay = np.array2string(np.round(self.delay_vec_sel[f]*norm_fact,4), precision=4, separator=', ')
            delay_tab = self.var_names[self.var_idx]+" : "+ str(formated_delay) + '\n'
        
            tab = pd.concat((tab,
                             pd.DataFrame({ "Case":case_name[f],\
                                            "Method":"AMI-FNN",\
                                            "Dimension":self.dim_sel[f], \
                                            "Delay ("+nf_str+")": [delay_tab]})))
        return tab
            

    def plot_ami_fnn(self, folder, save= False):
        for i in range(len(self.param_train)):
            f = self.param_train.iloc[i].datafile
            ami = self.ami[f]
            tau_ami = self.tau_ami[f]
            t = self.t[f]
            dim = self.dim[f]
            fnn = self.fnn[f]
            fnn_zero = self.fnn_zero[f]

            fig = plt.figure(figsize=(12,6))
    
            ax  = fig.add_subplot(1,2,1)

            ax.plot(t[:len(ami)],ami, c='red')
            ax.set_xlabel('$\\tau~(s)$'  , fontsize=20)
            ax.set_ylabel('$AMI~(bits)$' , fontsize=20)

            
            ax.plot(t[tau_ami],ami[tau_ami],'v', label='$\\tau='+str("{:.2e}".format(t[tau_ami]))+'='+str(tau_ami)+'\Delta t $')
            ax.legend(fontsize=20)
            
            ax  = fig.add_subplot(1,2,2)

            ax.plot(dim,fnn,'-ob')

            ax.plot(dim[fnn_zero],[0], 'vr', label='$Dimension='+str(dim[fnn_zero])+'$',markersize=10)
            ax.legend(fontsize=20)

            [ax.annotate('$'+str(int(fnn[i]))+'$',[dim[i],fnn[i]+500], fontsize=15) for i in range(len(fnn))]
            ax.set_xlabel('$Dimension$', fontsize=20)
            ax.set_ylabel('$FNN$', fontsize=20)
            #axs.set_xlim([min(arr_plot[:,0]),max(arr_plot[:,0])])

            fig.suptitle("$"+self.case_names[f]+"$", fontsize=25)

            fig.savefig('./'+folder+'/AMI_FNN_'+str(i)+'.pdf')

def get_fnn(x, dmax, tau, Rtol=10, Atol=2):
    eps = np.finfo(float).eps
    fnn = np.zeros(dmax)
    dim = []
    
    sigma = np.std(x)
    
    xzc = x-np.mean(x)              # zero centered x
    
    xc  = delayTS(xzc,tau,dim=1)    # current x delayed 
    
    for d in range(1,dmax+1):
        #print('d=',d)
        dim.append(d)
        
        xn = delayTS(xzc,tau,d+1)
        xc = xc[:xn.shape[0],:]
        
        for j in range(xc.shape[0]):
            dist = np.sum((xc - xc[j])**2, axis=1)
            
            id_so = np.argsort(dist)
            
            for id_se in range(1,5):
                try:
                    id_ne = id_so[id_se]
                    dc_ne = dist[id_ne]       # current distance of nearest neighbor

                    dn_ne = dc_ne+ (xn[id_ne,-1]-xn[j,-1])**2 

                    Rc    = np.sqrt(dc_ne) + eps
                    Rn    = np.sqrt(dn_ne) + eps

                    if np.sqrt(dn_ne-dc_ne)/Rc > Rtol:
                        fnn[d-1] += 1
                    elif (Rn/sigma)>Atol:
                        fnn[d-1] += 1
                    break
                except:
                    print('exception occured for j=',j)
                        
        xc = xn    
    return dim, fnn

def delayTS(x,tau,dim):
    xd = []
    s  = 0
    for i in range(dim):
        s  = i*tau 
        e  = len(x) - (dim-1-i)*tau
        #print('s=',s)
        #print('e=',e)
        if i == 0:
            xd = x[s:e].reshape(-1,1)
        else:
            xd = np.hstack((xd,x[s:e].reshape(-1,1)))
    return xd

def get_ami(xg, tau_max):
    dim = 2
    ami = np.zeros(tau_max)
    eps  = np.finfo(float).eps
    
    x    = (xg-min(xg))
    x    = x*(1-eps)/max(x)
    
    n_bins = np.array(np.ceil(np.log2(len(x))), dtype=int)#//5
    #print(n_bins)
    
    x    = np.array(np.floor(x*n_bins), dtype=int)
    
    for tau in range(tau_max):
        pxy      = np.zeros((n_bins,n_bins))
        #print(pxy)
        
        xd       = delayTS(x,tau,dim)
        #print(xd)
        
        for xt in xd:
            pxy[xt[0], xt[1]] +=1 
        
        pxy = pxy/xd.shape[0] + eps
        
        px  = np.sum(pxy, axis = 1)
        py  = np.sum(pxy, axis = 0)
        
        pd  = np.outer(px,py)
        temp     = pxy/pd
        temp     = pxy*np.log2(temp)
        ami[tau] = np.sum(temp.reshape(-1))
    return ami

def get_local_min(ami, win):    
    for i in range(win,len(ami)-win):
        if (ami[i-1]-ami[i])>0 and ami[i+1]-ami[i]>0:
            id_min = i
            #print(id_min)
            break

    tau_ami = id_min
    return tau_ami 

def ami_fnn(x, sampling_time, tau_max, fnn_threshold, win =2, dim_max = 15):
    t = np.arange(len(x))*sampling_time

    ami = get_ami(x, tau_max)
    tau_ami = get_local_min(ami, win)
    dim, fnn = get_fnn(x, dim_max, tau_ami)

    fnn_zero = np.where(fnn<fnn_threshold)[0][0]

    dim_sel = dim[fnn_zero]
    delay_vec_sel = np.arange(dim_sel)*t[tau_ami]
    return ami, tau_ami, dim, fnn, fnn_zero, dim_sel, delay_vec_sel