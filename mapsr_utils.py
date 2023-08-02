import sys
import numpy as np
import pandas as pd
import schemes_dev as sc
import torch
import torch.optim as optim
import json

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
        self.viz=False          # Visualization
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
                print(var+'='+ str(getattr(self,var)))


def get_args(param_df, i):
    args = model_param()
    column_name = list(param_df.keys())
    for var in column_name[1:]:
        if var=='lr_tau':
            exec('args.lr_τ=param_df.iloc['+str(i)+']["'+var+'"]')
        else:
            exec('args.'+var+'=param_df.iloc['+str(i)+']["'+var+'"]')
    return args

def get_Job_list(my_rank,n_Job,n_proc):
    Job_list = []
    for j in range(int(np.ceil(n_Job/n_proc))):
        Job_id = my_rank + n_proc*j
        if Job_id<n_Job:
            Job_list.append(Job_id)
    return Job_list

def expand_parameters(param_df_in):
    param_df = pd.DataFrame()
    column_name_in = list(param_df_in.keys())
    count = 0
    for i in range(len(param_df_in)):
        param_df_temp = pd.DataFrame()
        print("Dimension range = ")
        print(param_df_in.iloc[i]['dim_min'], end='-')
        print(param_df_in.iloc[i]['dim_max'])

        dim_min = param_df_in.iloc[i]['dim_min']
        dim_max = param_df_in.iloc[i]['dim_max']

        n =  dim_max - dim_min + 1

        for col in column_name_in:
            if col == 'Case':
                param_df_temp[col] = count + np.arange(n)
            elif col == 'folder_init': 
                param_df_temp['folder'] = [param_df_in.iloc[i][col]+'_D'+str(j) for j in range(dim_min,dim_max+1)]
                param_df_temp['dimensions'] = np.arange(dim_min, dim_max+1)
            else:
                param_df_temp[col] = [param_df_in.iloc[i][col]]*n
        count += n 
        param_df = pd.concat([param_df,param_df_temp])
    return param_df


# From mapsr.py
def set_device(args):
    cpu = torch.device('cpu')
    if args.use_gpu:
        if platform.system()=="Darwin":
            try:
                device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
            except:
                device = torch.device("cpu")        
        else:
            device = torch.device("cuda:" + str(args.gpu) if torch.cuda.is_available() else "cpu")
    else:
        device = cpu
    return device

def load_data(args, device, dtype):
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
    
    x_true = []

    print("Timeseries index (index starts with zero):",args.timeseries_index)

    for id in json.loads(args.timeseries_index):
        x_true.append(x_true_sam[:,id])

    dt = dt.to(device)

    return t_true, x_true, dt


# # Method to creates batch_size number of batches of true data of batch_time duration

def get_batch(t_true, x, Nc, τ_arr, batch_time, batch_size, device= torch.device('cpu')):
    t = t_true
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
def get_fun(args, func, optimizer, optimizer_τ, lr, lr_τ, τ_arr, acc):

    τ_all_1, W_all_1, b_all_1 = sc.merge_multi(τ_arr, func, acc)

    dim = 0
    for τ in τ_all_1:
        dim += len(τ)

    sys.path.append(args.folder)
    from neuralODE import ODEFunc

    func = ODEFunc(dimensions=dim) 
    sc.set_all_weight_mat(func, W_all_1)
    sc.set_all_bias_mat(func, b_all_1)
        
    optimizer = optim.RMSprop(func.parameters(), lr=lr)       # lr=1e-4
    optimizer_τ = optim.RMSprop(τ_all_1, lr=lr_τ)             # lr=1e-6 => Gives ok results
    
    return func, optimizer, optimizer_τ, τ_all_1

def initialize_delay_vector(x_true, args, device, dt):
    τ_arr = []
    for i in range(len(x_true)):
        τ = torch.linspace(0,(args.dimensions-1) *args.dtau * dt, args.dimensions)
        τ_arr.append(τ[0:args.dimensions].to(device).clone().detach().requires_grad_(True))
    dim = 0
    for τ in τ_arr:
        dim += len(τ)
    return τ_arr, dim

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

def restart(args, device, func, τ):
    print("\n"*3, "Loading state....")
    func.load_state_dict(torch.load(args.folder + '/Weight.pkl'))
    delay_temp = np.loadtxt(args.folder+'/delay.txt')
    if len(delay_temp.shape)==1:
        τ = torch.tensor([delay_temp[-1]]).to(device).clone().detach().requires_grad_(True)
    else:
        τ = torch.tensor(delay_temp[-1,:]).to(device).clone().detach().requires_grad_(True)
    print("Initial τ:",τ)
    print("Loading successful!"+"\n"*3)
    return func,τ

def get_sum(arr, power=1):
    s = torch.sum(torch.abs(arr[0])**power)
    for i in range(1,len(arr)):
        s = s + torch.sum(torch.abs(arr[i])**power)
    return s

def get_learning_rate(iteration, lr_start, lr_end, loc_trans, spread):
    lr =  lr_start*0.5*(1-np.tanh((iteration-loc_trans)/spread)) \
        + lr_end  *0.5*(1+np.tanh((iteration-loc_trans)/spread))
    return lr

def create_log(file, text):
    with open(file, 'w') as log_file:
        log_file.write(text+'\n')

def write_log(file, text):
    with open(file, 'a') as log_file:
        log_file.write(text +'\n')