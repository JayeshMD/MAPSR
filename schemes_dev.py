import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import pickle as pkl

def rk4(fun, t, x, h):
    k1 = fun(t, x)
    k2 = fun(t + h / 2, x + h * k1 / 2)
    k3 = fun(t + h / 2, x + h * k2 / 2)
    k4 = fun(t + h, x + h * k3)
    xn = x + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    return xn

def interp_linear(t, x, nc, τs, device= torch.device('cpu') ):
    flag = 0
    dt  = (t[1]-t[0]).to(device)
    id_list = torch.arange(nc).to(device)
    
    #print('Device of dt:', dt.device)
    #print('Device of id_list:', id_list.device)
    #print('Device of τs:', τs.device)
    #print('Device of t:', t.device)

    for τ in τs:
        #print('Device of τ:', τ.device)
        with torch.no_grad():
            id_l = id_list[(t[0:nc]>= τ-dt) * (t[0:nc]<= τ)]
            id_l = id_l[0]

        α = (τ-t[id_l])/dt
    
        z_temp = (1-α)*x[id_l:-(nc-id_l)] + α*x[id_l+1:-(nc-id_l-1)]
        z_temp = z_temp.reshape(-1,1)

        if flag==0:
            z = z_temp
            flag = 1
        else:
            z = torch.cat([z, z_temp], 1)
    return z
#=====================================================
class vector:
    def __init__(self,v, **kwargs):
        self.vector = v
        if('W_all' in kwargs.keys()):
            self.W_all = kwargs['W_all']
        else:
            self.W_all = [np.zeros((len(v),len(v)))]

        if('bias_all' in kwargs.keys()):
            self.bias_all = kwargs['bias_all']
        else:
            self.bias_all = [np.zeros(len(v))]

    def find_index_of(self, val):
        num = np.arange(len(self.vector))
        return num[self.vector==val]

    def find_index_within_delta(self, val, delta):
        num = np.arange(len(self.vector))
        return num[(self.vector<(val+abs(delta))) & 
                   (self.vector>(val-abs(delta)))]

    def get_index_of_neb(self, index, delta):
        index_neb = self.find_index_within_delta(self.vector[index], delta)
        index_neb = index_neb[index_neb!=index]
        return index_neb

    def remove_arond(self, index, delta):
        index_neb = self.get_index_of_neb(index, delta)
        self.vector = np.delete(self.vector, index_neb)
        for i in index_neb:
            self.W_all[0][:,index] += self.W_all[0][:,i]
        
        self.W_all[0]  = np.delete(self.W_all[0] , index_neb, 1)
        self.W_all[-1] = np.delete(self.W_all[-1], index_neb, 0)
        self.bias_all[-1] = np.delete(self.bias_all[-1], index_neb)

    def merge(self, delta):
        i = 0
        while(i<len(self.vector)):
            self.remove_arond(i,delta)
            i = i + 1

    def __repr__(self):
        return self.vector.__repr__()

def get_first_weight_mat(ODEfunc):
    for m in ODEfunc.net.modules():
        if isinstance(m, nn.Linear):
            return m.weight.data.detach().numpy()

def get_last_weight_mat(ODEfunc):
    count1 = 0 
    for m in ODEfunc.net.modules():
        if isinstance(m, nn.Linear):
            count1 +=1

    count2 = 0
    for m in ODEfunc.net.modules():
        if isinstance(m, nn.Linear):
            count2 +=1
            if (count2==count1):
                return m.weight.data.detach().numpy()

def set_first_weight_mat(ODEfunc, W):
    for m in ODEfunc.net.modules():
        if isinstance(m, nn.Linear):
            m.weight.data = torch.tensor(W, dtype = torch.float32)
            return

def set_last_weight_mat(ODEfunc, W):
    count1 = 0 
    for m in ODEfunc.net.modules():
        if isinstance(m, nn.Linear):
            count1 +=1

    count2 = 0
    for m in ODEfunc.net.modules():
        if isinstance(m, nn.Linear):
            count2 +=1
            if (count2==count1):
                m.weight.data = torch.tensor(W, dtype = torch.float32)
                return ODEfunc

#=================================================
def get_all_weight_mat(ODEfunc):
    W_all = []
    for m in ODEfunc.net.modules():
        if isinstance(m, nn.Linear):
            W_all.append(m.weight.data.detach().numpy())
    return W_all

def get_all_bias_mat(ODEfunc):
    b_all = []
    for name, params in ODEfunc.named_parameters():
        if 'bias' in name:
            b_all.append(params.detach().numpy())

    return b_all

def set_all_weight_mat(ODEfunc,W_mat):
    i = 0
    for m in ODEfunc.net.modules():
        if isinstance(m, nn.Linear):
            m.weight.data = torch.tensor(np.array(W_mat[i]), dtype = torch.float32)
            i += 1

def set_all_bias_mat(ODEfunc, bias_mat):
    i = 0
    for m in ODEfunc.net.modules():
        if isinstance(m, nn.Linear):
            m.bias = nn.Parameter(torch.tensor(bias_mat[i], dtype = torch.float32))
            i += 1


#=====================================================
def merge(v,acc):
    i = 0
    L = len(v)
    v = v[v>=0]
    while i<L:
        v0 = v[0:i] 
        v1 = v[i+1:]
        id0 = (v0-v[i]).abs()>acc
        id1 = (v1-v[i]).abs()>acc

        v0 = v0[id0]
        v1 = v1[id1]

        v  = torch.cat((v0,v[i].reshape(1,), v1),0)
        if L==len(v):
            i += 1
        L = len(v)
    return v

def merge_vw(v,W_in, W_out, acc):
    i = 0
    L = len(v)
    W_out = torch.t(W_out)
    while i<L:
        # print("Scheme i:=",i)
        v0 = v[0:i] 
        v1 = v[i+1:]

        Wr0, Wl0 = W_in[0:i,:] , W_out[0:i,:]
        Wr1, Wl1 = W_in[i+1:,:], W_out[i+1:,:]

        id0_sum = (v0-v[i]).abs()<=acc  # corresponding rows or colums are summed into i^{th} row or column based on NN implementation  
        id1_sum = (v1-v[i]).abs()<=acc  # corresponding rows or colums are summed into i^{th} row or column based on NN implementation  
        
            
        id0 = (v0-v[i]).abs()>acc      
        id1 = (v1-v[i]).abs()>acc
        
        # for v
        v0 = v0[id0]
        v1 = v1[id1]
        v  = torch.cat((v0,v[i].reshape(1,), v1),0)
        
        # for W
        # print("torch.sum(W_in[id0_sum,:],axis=0)=",id0_sum)
        # print("id0_sum:", sum(id0_sum))
        # print("id1_sum:", sum(id1_sum))

        # print("id0_cmp:", sum(id0_sum)>0)
        # print("id1_cmp:", sum(id1_sum)>0)

        if sum(id0_sum)>0:
            W_in[i,:] += torch.sum(Wr0[id0_sum,:],axis=0)
        if sum(id1_sum)>0:    
            W_in[i,:] += torch.sum(Wr1[id1_sum,:],axis=0)
     
        Wr0 = Wr0[id0,:]
        Wr1 = Wr1[id1,:]

        Wl0 = Wl0[id0,:]
        Wl1 = Wl1[id1,:]

        W_in   = torch.cat((Wr0,W_in[i,:].reshape(1,-1), Wr1),0)
        W_out  = torch.cat((Wl0,W_out[i,:].reshape(1,-1), Wl1),0)        

        if L==len(v):
            i += 1
        L = len(v)
    return v, W_in, torch.t(W_out)


def merge_vw_bkp(v,W,acc):
    i = 0
    L = len(v)
    while i<L:
        v0 = v[0:i] 
        v1 = v[i+1:]
        id0 = (v0-v[i]).abs()>acc
        id1 = (v1-v[i]).abs()>acc
        
        # for v
        v0 = v0[id0]
        v1 = v1[id1]
        v  = torch.cat((v0,v[i].reshape(1,), v1),0)
        
        # for W
        Wr0 = W[0:i,:]
        Wr1 = W[i+1:,:]

        Wr0 = Wr0[id0,:]
        Wr1 = Wr1[id1,:]
        W   = torch.cat((Wr0,W[i,:].reshape(1,-1), Wr1),0)

        Wc0 = W[:,0:i]
        Wc1 = W[:,i+1:]

        Wc0 = Wc0[:,id0]
        Wc1 = Wc1[:,id1]

        W   = torch.cat((Wc0,W[:,i].reshape(-1,1), Wc1),1)

        if L==len(v):
            i += 1
        L = len(v)
    return v, W

def get_arr_sum(d,s,l):
    # d = dimension, s = sum, l = generally empty list or list to which we want to append result
    if d==1:
        l.append([s])
        return l
    else:
        for i in range(s+1):
            l_temp = []
            get_arr_sum(d-1, s-i, l_temp)
            l += [ [i] + lt for lt in l_temp] 
        return l

def get_pow_mat(dim, degree):
    pow_list = []
    for s in range(degree+1):
        get_arr_sum(dim, s, pow_list)
    return pow_list

def get_feature_mat(x, degree): 
    pow_mat = torch.tensor(get_pow_mat(dim=x.shape[1], degree=degree))
    #print(pow_mat)
    flag = True
    for pm in pow_mat:
        xt = x.pow(pm)
        xm = xt[:,0]
        for i in range(1,xt.shape[1]):
            xm *= xt[:,i]
        if flag:
            A = xm.reshape([-1,1])
            flag = False
        else:
            #print(A.shape)
            A = torch.cat([A,xm.reshape(-1,1)], 1)
    return A

# %%
def avg_mutual_info(x,lag_max):
    """
    avg_mutual_info(x,lag_max)
    """

    eps = np.finfo(float).eps #float epsilon or machine precision

    x = x-x.min()
    x = x/x.max()
    
    N = len(x)
    L = np.arange(lag_max)
    I = np.zeros((len(L),1))
    
    for i in L:
        bins = int(np.ceil(np.log2(N-L[i])))

        bing = np.floor(x*bins)
        binx = bing[0:(N-L[i])]
        biny = bing[L[i]:]

        P = np.zeros((bins,bins))

        for j in range(bins):
            idx  = (binx==j)
            for k in range(bins):
                idxy = (biny[idx]==k)
                P[j,k] = sum(idxy)      

        P  = P/(N-L[i])
        Px = np.sum(P,axis=0).reshape(-1,1)
        Py = np.sum(P,axis=1).reshape(-1,1)

        P  = P  + eps
        Px = Px + eps
        Py = Py + eps

        I[i] = np.sum( P * np.log2(P/ np.dot(Px,Py.transpose())))

    return I

def set_delay_time_series(X, n, tau):
    data_delay = []
    for i in range(n):
        data_delay.append((X[(i*tau):(len(X)-(n-i-1)*tau)]))
    data_delay = np.array(data_delay).squeeze().transpose()
    return data_delay.reshape(-1,n)


    # %%
if __name__ == '__main__':
    x       = dat.database.get('p')
    lag_max = 20
    I     = avg_mutual_info(x,lag_max)  


    # %%
    for i in range(len(I)-1):
        if (I[i]<I[i-1]) * (I[i]<I[i+1]):
            tau_ami = i
            break
            
    tau_ami


    # %%
    plt.figure()
    plt.plot(I,'-x')


    # %%
    dat.set_delay_timeseries(var_name='p', n=3 ,tau=tau_ami)
    plt.figure()
    plt.plot(dat.data_delay[:,0],dat.data_delay[:,1])


    # %%
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.plot3D(dat.data_delay[:,0],dat.data_delay[:,1],dat.data_delay[:,2])


    # %%
    x = dat.database.get('p')


    # %%
    plt.figure()
    plt.plot(dat.database.t,x)


# %%
def false_nearest_neighbour(X,d_max,tau,Rtol=10, Atol=2):
    
    x = X - np.mean(X)
    N = len(x)
    FNN = np.zeros(d_max)
    
    for dim in range(1,d_max): # gives the dimention of the present phase space 
        print('Dim='+str(dim))

        Y   = set_delay_time_series(X=x[:N-(dim)*tau], n=dim, tau = tau )
        N1  = Y.shape[0]

        print(Y.shape)

        for j in range(N1):

            R   = np.sqrt(np.sum((Y-Y[j])**2, axis=1))

            ids = np.argsort(R)

            Rd1 = R[ids[1]]
            
            Rd12 = (x[j+ dim*tau] - x[ids[1]+ dim*tau])  # Distance added to nearest neighbour
            Rd2  = np.sqrt(Rd1**2 + Rd12**2)

            if (abs(Rd12)/Rd1) > Rtol:
                FNN[dim] = FNN[dim]+1

            elif Rd2/np.std(x)> Atol:
                FNN[dim] = FNN[dim]+1

    FNN = (FNN/FNN[1])* 100
    
    return FNN


#==============================================

def interp_linear_multi(t_arr, x_arr, nc, τs, device= torch.device('cpu') ):
    # print("interp_linear_multi:")
    # print("\tsize(τs) = ", len(τs))
    n = len(τs)
    z = interp_linear(t_arr[0], x_arr[0], nc, τs[0], device=device)
    for i in range(1,n):
        z = torch.cat( [z, interp_linear(t_arr[i], x_arr[i], nc, τs[i], device=device)], 1)    
    return z

def merge_multi(τ_arr, func, acc):
    W_all_0 = get_all_weight_mat(func)
    b_all_0 = get_all_bias_mat(func)

    # print(W_all_0)
    # print(b_all_0)

    τ_all_1 = []
    col_start = 0
    
    for i in range(len(τ_arr)):
        l_0 = len(τ_arr[i])
        col_end   = col_start + l_0

        W_all = [W_all_0[0][:,col_start:col_end]]
        W_all += W_all_0[1:-1]                          # not a normal addition appends W_all_0[1:-1] to W_all
        W_all.append(W_all_0[-1][col_start:col_end,:])

        b_all = b_all_0[0:-1]
        b_all.append(b_all_0[-1][col_start:col_end])
        
        x = vector(τ_arr[i].detach().numpy(), W_all = W_all, bias_all = b_all)
        x.merge(delta=acc.numpy())
        
        if(i==0):
            W_all_1 = x.W_all
            b_all_1 = x.bias_all
        else:
            W_all_1[0]  = np.hstack([ W_all_1[0], x.W_all[0]])
            W_all_1[-1] = np.vstack([ W_all_1[-1], x.W_all[-1]])

            b_all_1[-1] = np.hstack([b_all_1[-1], x.bias_all[-1]])

        τ_all_1.append(torch.tensor(x.vector).requires_grad_(True))
        col_start = col_end

    return τ_all_1, W_all_1, b_all_1
    
class model_param:
    def __init__(self):
        self.adjoint = False    # Method of integration
        self.batch_time=30     # Time duration of training batch in terms of Δt steps
        self.batch_size=30     # Number of batches of training data
        self.lr = 1e-4         # Learning rate for model parameters
        self.lr_τ = 1e-5       # Leaning rate delay vector τ
        self.Nc = 40         # Maximum delay in terms of Δt steps
        self.niters=2000       # Maximum number of iterations
        self.test_freq=1       # Testing frequency in terms of iterations (for print and plot)
        self.viz=True          # Visualization
        self.savefig= True     # Set True to save figure
        self.folder='output_folder'   # folder name to store output
        self.use_gpu = False
        self.restart = False
        self.gpu = 0
        self.datafile = "path_to_data_file.txt"
        self.dimensions = 4
        
    def to(self, device):
        args_dir = dir(self)
        for var in args_dir:
            if type(getattr(self, var)) == int or type(getattr(self, var)) == int:
                exec('self.'+var+'=torch.tensor(self.'+var+').to(device)')
                #print(var+'='+ str(getattr(self,var)))


