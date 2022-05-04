import torch

def rk4(fun, t, x, h):
    k1 = fun(t, x)
    k2 = fun(t + h / 2, x + h * k1 / 2)
    k3 = fun(t + h / 2, x + h * k2 / 2)
    k4 = fun(t + h, x + h * k3)
    xn = x + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    return xn


def interp_linear(t, x, nc, τs):
    flag = 0
    for τ in τs:
        dt  = t[1]-t[0]
        τ_l = t[0:nc-1]
        τ_u = t[1:nc]

        α = (τ >= τ_l) * (τ < τ_u)
        for i in range(len(x) - nc):
            A = torch.cat([x[i:i+nc-1].reshape(-1, 1), x[i+1:i+nc].reshape(-1, 1)], 1)
            P = (α/dt).reshape(-1, 1) * torch.cat([(τ_u-τ).reshape(-1, 1), (τ - τ_l).reshape(-1, 1)], 1)
            z2 = (A * P).sum()
            if i == 0:
                z_temp = z2.reshape(-1, 1)
            else:
                z_temp = torch.cat([z_temp, z2.reshape(-1, 1)], 0)
        if flag==0:
            z = z_temp
            flag = 1
        else:
            z = torch.cat([z, z_temp], 1)
    return z

def merge(v,acc):
    i = 0
    L = len(v)
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

def merge_vw(v,W,acc):
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