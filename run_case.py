#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import mapsr as psr
from mpi4py import MPI
from mpi4py.util import dtlib
import time

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
                print(var+'='+ str(getattr(self,var)))

# In[3]:


param_df = pd.read_csv("Lorenz/Param.csv")
column_name = list(param_df.keys())
# In[4]:


def fun_run(i):
    args = model_param()
    for var in column_name[1:]:
        if var=='lr_tau':
            exec('args.lr_τ=param_df.iloc['+str(i)+']["'+var+'"]')
        else:
            exec('args.'+var+'=param_df.iloc['+str(i)+']["'+var+'"]')
    psr.mapsr(args)


# In[5]:
datatype = MPI.INT
np_dtype = dtlib.to_numpy_dtype(datatype)
itemsize = datatype.Get_size()

N = 1
win_size = N * itemsize if my_rank == 0 else 0
win = MPI.Win.Allocate(win_size, comm=comm)

buf = np.zeros(N, dtype=np_dtype)
win.Lock(rank=0)
win.Put(buf, target_rank=0)
win.Unlock(rank=0)

N_Jobs = len(param_df)

if my_rank == 0:
    print('Number of cases:', N_Jobs)

while buf[0]<N_Jobs:
    win.Lock(rank=0)
    win.Get(buf, target_rank=0)
    buf[0] += 1
    win.Put(buf, target_rank=0)
    win.Unlock(rank=0)
    if buf[0]<N_Jobs:
        Job_Sel = buf[0]
        print(param_df.iloc[Job_Sel])
        print("Rank:", my_rank, " takes the Job:", Job_Sel)
        fun_run(Job_Sel)