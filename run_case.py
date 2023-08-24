#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import mapsr as psr
from mpi4py import MPI
import time
import numpy as np

# From mapsr
import mapsr_utils

comm    = MPI.COMM_WORLD # Communicator object
my_rank = comm.Get_rank()
n_proc  = comm.Get_size()     # Number of parallel process i.e. number of processors requested during mpirun

print("My_rank:", my_rank)

param_df_in    = pd.read_csv("Lorenz/Param.csv")
column_name_in = list(param_df_in.keys())
param_df       = mapsr_utils.expand_parameters(param_df_in)
column_name    = list(param_df.keys())
Job_list       = mapsr_utils.get_Job_list(my_rank,len(param_df),n_proc)

for i in Job_list:
    args = mapsr_utils.get_args(param_df, i)
    psr.mapsr(args)


