import torch
import numpy as np

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

def interp_linear_multi(t_arr, x_arr, nc, τs, device= torch.device('cpu') ):
    n = len(τs)
    z_arr = []
    for i in range(n):
        z = interp_linear(t_arr[i], x_arr[i], nc, τs[i])
        z_arr.append(z)    
    return z_arr


class vector:
    def __init__(self,v, **kwargs):
        self.vector = v
        if('W_in' in kwargs.keys()):
            self.matrix_in = kwargs['W_in']
        else:
            self.matrix_in = np.zeros((len(v),len(v)))

        if('W_out' in kwargs.keys()):
            self.matrix_out = kwargs['W_out']
        else:
            self.matrix_out = np.zeros((len(v),len(v)))

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
            self.matrix_in[:,index] += self.matrix_in[:,i]
        self.matrix_in = np.delete(self.matrix_in, index_neb, 1)
        self.matrix_out = np.delete(self.matrix_out, index_neb, 0)

    def merge(self, delta):
        i = 0
        while(i<len(self.vector)):
            self.remove_arond(i,delta)
            i = i + 1

    def __repr__(self):
        return self.vector.__repr__()

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