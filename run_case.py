#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import mapsr as psr
from mpi4py import MPI
import time
import numpy as np

comm = MPI.COMM_WORLD # Communicator object
my_rank = comm.Get_rank()
n_proc = comm.Get_size()     # Number of parallel process i.e. number of processors requested during mpirun

print("My_rank:", my_rank)

# In[2]:


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

# In[3]:


param_df_in = pd.read_csv("Exp_Tara/Param.csv")
column_name_in = list(param_df_in.keys())
# In[4]:

print(param_df_in)
print(column_name_in)

param_df = pd.DataFrame()
count = 0
for i in range(len(param_df_in)):
    param_df_temp = pd.DataFrame()
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

column_name = list(param_df.keys())
# print(param_df)
# print(len(param_df))
# param_df = param_df.set_index(np.arange(len(param_df)))
# print(param_df)
# exit()

def fun_run(i):
    args = model_param()
    for var in column_name[1:]:
        if var=='lr_tau':
            exec('args.lr_τ=param_df.iloc['+str(i)+']["'+var+'"]')
        else:
            exec('args.'+var+'=param_df.iloc['+str(i)+']["'+var+'"]')
    psr.mapsr(args, comm)


# In[5]:
def get_Job_list(my_rank,n_Job,n_proc):
    Job_list = []
    for j in range(int(np.ceil(n_Job/n_proc))):
        Job_id = my_rank + n_proc*j
        if Job_id<n_Job:
            Job_list.append(Job_id)
    return Job_list

Job_list = get_Job_list(my_rank,len(param_df),n_proc)

for i in Job_list:
    # try:
    print(param_df.iloc[i])
    fun_run(i)
    # except:
    #     pass
