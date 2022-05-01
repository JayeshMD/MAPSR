import torch

def rk4(fun, t, x, h):
    k1 = fun(t, x)
    k2 = fun(t + h / 2, x + h * k1 / 2)
    k3 = fun(t + h / 2, x + h * k2 / 2)
    k4 = fun(t + h, x + h * k3)
    xn = x + (h / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    return xn


def interp_linear(t, x, nc, τ_arr):
    if τ_arr.shape[0] > 1:
        print("Only one tau is supported.")
    for τ in τ_arr:
        dt  = t[1]-t[0]
        τ_l = t[0:nc-1]
        τ_u = t[1:nc]

        α = (τ >= τ_l) * (τ < τ_u)
        for i in range(len(x) - nc):
            A = torch.cat([x[i:i+nc-1].reshape(-1, 1), x[i+1:i+nc].reshape(-1, 1)], 1)
            P = (α/dt).reshape(-1, 1) * torch.cat([(τ - τ_l).reshape(-1, 1), (τ_u - τ).reshape(-1, 1)], 1)
            z2 = (A * P).sum()
            if i == 0:
                z_true = torch.cat([x[i].reshape(-1, 1), z2.reshape(-1, 1)], 1)
            else:
                z_temp = torch.cat([x[i].reshape(-1, 1), z2.reshape(-1, 1)], 1)
                z_true = torch.cat([z_true, z_temp], 0)
    return z_true