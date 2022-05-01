import torch
import matplotlib.pyplot as plt
from schemes import Rk4
# Data type and device setup
dtype = torch.float
device = torch.device("cpu")

# Generate signal
freq = 1
t_true = torch.linspace(0, 10, 1001)
x_true = torch.sin(2 * torch.pi * freq * t_true)

fig = plt.figure()
axs = fig.add_subplot(1, 1, 1)
axs.plot(t_true, x_true)

plt.show()

dt = t_true[1] - t_true[0]
Nc = 100


##



##
def model_torch(t, xx, w0):
    return torch.mm(xx, w0)


##
τ = torch.tensor(0.2, device=device, dtype=dtype, requires_grad=True)
# w0 = torch.randn((2,2), device=device, dtype=dtype, requires_grad=True)
w0 = torch.zeros((2, 2), device=device, dtype=dtype, requires_grad=True)
# a = torch.randn((2,2), dtype=dtype).abs()*1e-5
# b = torch.normal(mean=a*0., std=a)
# w0 = b.clone().detach().requires_grad_(True).to(device)

# w = [[0,1],[-(2*torch.pi)**2, 0]]
# w0 = torch.tensor(w, device=device, dtype=dtype, requires_grad=True)
# w0 = torch.tensor([[0, 6.28],[-6.28,0]], device=device, dtype=dtype, requires_grad=True)

##
lr_tau = 1e-4
lr_w0 = 1
batch_size = 20
batch_time = 10  # (x_true)-Nc
lr_pow_w0 = torch.linspace(0, -2, 100)
lr_pow_tau = torch.linspace(0, -4, 100)
##
loss_arr = []
τ_arr = []
τ_grad_arr = []

##
fig = plt.figure()

for kk in range(2000):

    # lr_w0 = 1e-4 + 1e-1 * torch.tanh(torch.tensor(kk/500))
    # lr_tau = 1e-1 * torch.tensor(10).pow(lr_pow_tau[kk//100])
    batch_time = 10 + ((kk // 100) * 10);

    if kk % 100 == 0:
        print('lr_w0=', lr_w0)
        print('batch_time=', batch_time)

    τ_l = t_true[0:Nc - 1]
    τ_u = t_true[1:Nc]

    α = (τ >= τ_l) * (τ < τ_u)
    for i in range(len(x_true) - Nc):
        A = torch.cat([x_true[i:i + Nc - 1].reshape(-1, 1), x_true[i + 1:i + Nc].reshape(-1, 1)], 1)
        P = (α / dt).reshape(-1, 1) * torch.cat([(τ - τ_l).reshape(-1, 1), (τ_u - τ).reshape(-1, 1)], 1)
        z2 = (A * P).sum()
        if i == 0:
            z_true = torch.cat([x_true[i].reshape(-1, 1), z2.reshape(-1, 1)], 1)
        else:
            z_temp = torch.cat([x_true[i].reshape(-1, 1), z2.reshape(-1, 1)], 1)
            z_true = torch.cat([z_true, z_temp], 0)

    if kk % 100 == 0:
        st_id = torch.randint(0, len(z_true) - batch_time, (1,)).item()

    id_sel = torch.randint(0, z_true.shape[0] - batch_time, (batch_size,))
    z_true_stack = torch.stack([z_true[id_sel + i, :] for i in range(batch_time)], dim=0)
    t_true_stack = torch.stack([t_true[id_sel + i] for i in range(batch_time)], dim=0)
    # print(z_true_stack.shape)
    # print(t_true_stack.shape)

    for i in range(0, batch_time):
        fun = lambda t, x: model_torch(t, x, w0)
        if i == 0:
            z_pred = z_true_stack[i, :, :].reshape(1, z_true_stack.shape[1], z_true_stack.shape[2])
        else:
            z_next = Rk4(fun, t_true[i], z_pred[i - 1, :, :], dt)
            z_pred = torch.cat([z_pred, z_next.reshape(1, z_true_stack.shape[1], z_true_stack.shape[2])], 0)

    # print(z_pred.shape)

    # print("z_pred.shape=", z_pred.shape)
    # print("z_true.shape=",z_true[0:batch_time,:].shape)
    # loss = torch.abs((z_true[0:batch_time,:]-z_pred)).sum() + torch.abs(w0).sum()

    # loss = (z_true_stack-z_pred).pow(2).mean() #+ 1e-5* torch.abs(w0).sum()
    loss = torch.abs((z_true_stack - z_pred)).mean() + 1e-2 * torch.abs(w0).sum()
    loss.backward()

    with torch.no_grad():
        w0_old = w0.detach().numpy()
        τ_old = τ.detach().numpy()

        # print(τ_old)
        τ_arr.append(τ_old)
        τ_grad_arr.append(τ.grad.detach().numpy())
        # print(w0.grad)
        w0 -= lr_w0 * w0.grad
        if kk > 300:
            τ -= lr_tau * τ.grad * (10 * (kk % 10 == 0) + (kk % 10 != 0))

        loss_arr.append(loss.item())
        # print(loss.item())
        if kk % 10 == 0:
            axs = fig.add_subplot(1, 1, 1)
            for p_id in range(batch_size):
                axs.plot(t_true_stack.detach().numpy()[:, p_id], z_pred[:, p_id, 0].detach().numpy(), 'ro')
                axs.plot(t_true_stack.detach().numpy()[:, p_id], z_pred[:, p_id, 1].detach().numpy(), 'bo')

            axs.plot(t_true[0:len(z_true)], z_true[:, 0].detach().numpy(), 'r-')
            axs.plot(t_true[0:len(z_true)], z_true[:, 1].detach().numpy(), 'b-')

            plt.show(block=False)
            plt.draw()
            plt.pause(0.001)

        if kk % 25 == 0:
            print("iter=", kk)
            print("loss=", loss.detach().numpy())
            print("w0=", w0.detach().numpy())
            print("tau=", τ.detach().numpy())

        w0.grad = None
        τ.grad = None