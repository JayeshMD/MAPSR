import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
import sys
import mapsr
import mapsr_utils
import json

import schemes_dev as sc
import ffnn
import pandas as pd
import pickle as pkl
import torch
from torchdiffeq import odeint
import os

plt.rcParams.update({
    "text.usetex": True,
    "font.family": "sans-serif",
    "font.sans-serif": "Helvetica",
})


class pp_mapsr:
    def __init__(self, param_file, var_names = [], case_names = []):
        self.param_file = param_file
        self.param_df, self.args_all, self.loss_all, self.delay_all = self.get_train_details(param_file)
        self.loss_average = None
        self.delay_details = None
        self.var_names = var_names
        self.n_var = len(self.delay_all[0][0])
        self.case_names = case_names

    def get_time_series_index(self):
        return self.param_df.timeseries_index.iloc[0]


    def set_var_names(self, var_names):
        self.var_names = var_names
        

    def set_case_names(self, case_names):
        self.case_names = {}
        #set_name = input("Do you want to name cases?(y/n):").lower()
        set_name = 'y'
        count = 0 
        for key in self.get_data_files(): 
            if set_name=='y':
                #case_name[key] = input(key+':')
                self.case_names[key] = case_names[count]
            else:
                self.case_names[key] = key 
            count +=1

    def df_all_dim(self, param_df_in):
        '''
        The df_all_dim(param_df_in) takes param_df_in as input
        which cotains dimension range i.e. [dim_min, dim_max]. 
        The df_all_dim(param_df_in) expands this range and returns 
        new dataframe with each row for seperate dimension.
        '''
        column_name_in = list(param_df_in.keys())
        param_df = pd.DataFrame()
        count = 0
        for i in range(len(param_df_in)):
            param_df_temp = pd.DataFrame()

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

    def fun_run(self, param_df, i):
        args = sc.model_param()
        column_name = list(param_df.keys())

        for var in column_name[1:]:
            if var=='lr_tau':
                exec('args.lr_τ=param_df.iloc['+str(i)+']["'+var+'"]')
            else:
                exec('args.'+var+'=param_df.iloc['+str(i)+']["'+var+'"]')
        return args

    def get_train_details(self, param_file):
        args_all = []
        loss_all = []
        delay_all = []

        param_df_in = pd.read_csv(param_file)
        param_df = self.df_all_dim(param_df_in) 
        for i in range(len(param_df)):
            args_all.append(self.fun_run(param_df, i))
            folder = args_all[i].folder
            with open(folder+'/loss.pkl','rb') as fl:
                loss_all.append(pkl.load(fl))
                
            with open(folder+'/delay.pkl','rb') as fl:
                delay_all.append(pkl.load(fl))

        return param_df, args_all, loss_all, delay_all

    def group_delays(self, arr_in):
        #arr_in = arr_plot[1]
        #arr_in[iter][time_series_id] gives the list of dealy values at iter iteration for time series with time_series_id 
        
        n_var = len(arr_in[0])    # number of time series
        
        arr_out_all = []
        for k in range(n_var):
            arr_out = []

            id_arr = [0]
            temp_arr = [arr_in[0][k]]
        
            l_temp = len(arr_in[0][k])

            for i in range(1, len(arr_in)):
                if l_temp==len(arr_in[i][k]):
                    
                    id_arr.append(i)
                    temp_arr.append(arr_in[i][k])
                else:
                    arr_out.append([id_arr, np.array(temp_arr)])
                    id_arr = [i]
                    temp_arr = [arr_in[i][k]]
                    l_temp = len(arr_in[i][k]) 

            sz = np.array([len(t) for t in temp_arr])
            arr_out.append([id_arr, np.array(temp_arr)])
            arr_out_all.append(arr_out.copy())
        return arr_out_all

    def plot_delay(self, axs, arr_in, sym, color, markersize=5, markevery=1, fillstyle = 'none', label=None):
        arr_out_all = self.group_delays(arr_in)

        time_series_id = 0

        for arr_out in arr_out_all:
            count = 0
            for arr in arr_out:
                delay_data = arr[1]    #.squeeze(axis=1)
                
                for i in range(delay_data.shape[1]):
                    if count ==0:
                        axs[time_series_id].plot(np.array(arr[0]), delay_data[:,i], sym, color=color, markersize=markersize,
                                markevery=markevery, fillstyle = fillstyle,
                                label=label)
                        count +=1
                    else:
                        axs[time_series_id].plot(np.array(arr[0]),delay_data[:,i], sym, color = color, markersize=markersize,
                                markevery=markevery, fillstyle = fillstyle)
            time_series_id +=1
        return axs
    
    def get_data_files(self):
        temp = dict()
        args_all = self.args_all
        for i in range(len(args_all)):
            temp[args_all[i].datafile] = []

        catagory_keys = list(temp.keys())
        return np.array(catagory_keys)
    
    def set_loss_average(self, n=100):
        '''
        get_loss_average(self, n=100):
        Averages the loss for the last n iterations and stores as,\\
        >> self.loss_average[self.args_all[i].datafile].append([dim, loss_avg])\\
        Here, dim is the inital dimension for individual time series.
        '''
        self.loss_average = dict()
        for i in range(len(self.param_df)):
            loss_avg = np.mean(self.loss_all[i][-n:])
            dim = self.args_all[i].__dict__['dimensions']
            if not(self.args_all[i].datafile in self.loss_average.keys()):
                self.loss_average[self.args_all[i].datafile] = []
            
            self.loss_average[self.args_all[i].datafile].append([dim, loss_avg])

    def get_min_loss_idx(self):
        min_idx = dict()
        keys = self.get_data_files()
        for key in keys:
            arr_plot = np.array(self.loss_average[key])
            idx = np.where(arr_plot[:,1]==min(arr_plot[:,1]))[0][0]
            min_idx[key] = idx
        return min_idx

    def set_delay_details(self):
        delay_catagory = dict()

        param_df = self.param_df
        args_all = self.args_all
        delay_all = self.delay_all
        for i in range(len(args_all)):
            delay_catagory[args_all[i].datafile] = []
                
        for i in range(len(param_df)):
            dim = args_all[i].__dict__['dimensions']
            
            delay_catagory[args_all[i].datafile].append([dim, delay_all[i]])
        self.delay_details = delay_catagory

    def plot_mapsr_results(self, fig, axs, title_legend, ncol_legend = 2):
        if len(axs.shape)==1:
            axs = axs.reshape(-1,2)
        catagory = self.loss_average
        catagory_legend = self.case_names
        delay_catagory = self.delay_details
        n_var = len(self.delay_all[0][0])

        # Loss
        gs = axs[0, 0].get_gridspec()
        for ax in axs[0:, 0]:
            ax.remove()

        axs_big = fig.add_subplot(gs[0:,0])

        # Delays
        axs_right = axs[:,1]

        markersize=8

        #
        #  Loss
        #
        axs = axs_big

        color = ['forestgreen','b','deeppink','goldenrod','darkmagenta','r']
        sym   = ['-','--', '--o', '--s', '--x',':*']

        count = 0
        timeseries_id = 0

        color_count = 0

        for key in self.get_data_files():
            
            arr_plot = np.array(catagory[key])
            x = arr_plot[:,0]
            y = np.log10(arr_plot[:,1])

            axs.plot(x, y, sym[count%6], color=color[color_count],  
                    markersize=markersize,label="$"+catagory_legend[key]+"$", fillstyle = 'none')
            
            idx = np.where(arr_plot[:,1]==min(arr_plot[:,1]))[0][0]
            axs.plot(arr_plot[idx,0], np.log10(arr_plot[idx,1]), '*r', markersize=15)

            for case_id in range(len(delay_catagory[key])):
                    
                delay_end_t = delay_catagory[key][case_id][1][-1]
                dim_arr_end = [len(dd) for dd in delay_end_t]

                dim_end   = np.sum(dim_arr_end)
                
                dim_arr_end_str = np.array(dim_arr_end,dtype=str)
                dim_arr_end_label = '+'.join(dim_arr_end_str)

                if case_id==idx:
                    if self.n_var>1:
                        axs.text(x[case_id]-0.1, y[case_id]-0.12, "$("+ dim_arr_end_label+"="+str(dim_end)+")$", fontsize=20)
                    else:
                        axs.text(x[case_id]-0.1, y[case_id]-0.12, "$("+str(dim_end)+")$", fontsize=20)

            
            # For annotation: Final dimension for minimum loss case
            dim_end   = len(delay_catagory[key][idx][1][-1][timeseries_id])

            count += 1
            color_count +=1
            
            
        axs.xaxis.set_major_locator(tck.MultipleLocator(1))    
        axs.set_xlabel('$d_{init}$', fontsize=20)
        axs.set_ylabel('$\log_{10}(\\bar{\mathcal{L}}$)', fontsize=20)
        axs.set_xlim([min(arr_plot[:,0]),max(arr_plot[:,0])])

        axs.legend(title='$'+title_legend+'$', \
                ncol = ncol_legend, fontsize=18, title_fontsize=18, frameon=False, loc = 'upper left')

        axs.set_title("$(a)$", fontsize=20, loc='left')
        axs.tick_params(axis='both', which='major', labelsize=20)
        axs.tick_params(axis='both', which='minor', labelsize=20)


        #
        #  Delay
        #

        axs = axs_right

        count = 0 
        color_count = 0 

        min_idx = self.get_min_loss_idx()

        for key in self.get_data_files():
            arr_plot = delay_catagory[key][min_idx[key]]
            dim = arr_plot[0]

            self.plot_delay( axs, arr_plot[1], 
                            sym[count],
                            color[color_count],
                            markersize=markersize,
                            markevery=1000, 
                            fillstyle = 'none',
                            label=catagory_legend[key])
            
            count +=1
            color_count +=1

        title_ascii = ord('b')
        title_description = self.var_names

        for i in range(n_var):
            axs[i].set_xlabel('$Iterations$', fontsize=20)
            axs[i].set_ylabel('$\\tau$', fontsize=30)

            axs[i].set_xlim([0,len(arr_plot[1])])

            axs[i].tick_params(axis='both', which='major', labelsize=20)
            axs[i].tick_params(axis='both', which='minor', labelsize=20)

            axs[i].set_title("$("+chr(title_ascii)+")\:For\:time\:series\: of \:"+ title_description[i] +"$",fontsize=20, loc='left')
            title_ascii +=1

        plt.tight_layout()
        return axs_big, axs_right

    def create_table(self, norm_fact):


        catagory = self.loss_average
        delay_catagory = self.delay_details
        nf_str = "$\\times10^{"+str(int(np.log10(norm_fact)))+"}$"

        tab = pd.DataFrame(columns = ["Case", "Method", "Dimension", "Delay ("+nf_str+")"])
        tab_sup = pd.DataFrame(columns = ["Case", "Method", "Dimension", "Delay"])

        #
        #  Loss
        #
        count = 0

        for key in self.get_data_files():

            arr_plot = np.array(catagory[key])
            
            idx = np.where(arr_plot[:,1]==min(arr_plot[:,1]))[0][0]
    

            for case_id in range(len(delay_catagory[key])):
                
                delay_end_t = delay_catagory[key][case_id][1][-1]
                dim_arr_end = [len(dd) for dd in delay_end_t]

                dim_end   = np.sum(dim_arr_end)

                if case_id==idx:
                    
                    delay_tab = ''
                    for dd in range(len(delay_end_t)):
                        formated_delay = np.array2string(np.round(delay_end_t[dd]*norm_fact,4), precision=4, separator=', ')
                        delay_tab += self.var_names[dd]+" : "+ str(formated_delay) + '\n'

                    tab = pd.concat((tab,
                                    pd.DataFrame({ "Case":self.case_names[key],\
                                                    "Method":"MAPSR",\
                                                    "Dimension":dim_end, \
                                                    "Delay ("+nf_str+")": [delay_tab]})))
                    
                    delay_tab = []
                    for dd in range(len(delay_end_t)):
                        formated_delay = delay_end_t[dd]
                        delay_tab.append(formated_delay)

                    tab_sup = pd.concat((tab_sup,
                                    pd.DataFrame({  "Case":self.case_names[key],\
                                                    "Method":"MAPSR",\
                                                    "Dimension":dim_end, \
                                                    "Delay": [delay_tab]})))

            count += 1
        return tab, tab_sup
                    


def plot_cmp_2(axs, title, kk, tt,zt, tp,zp, τ, dpi = 300, col_1='k', col_2='r', delay_id = None, axis_labels=None):
    cpu = torch.device('cpu')

    right = 0.6
    top   = 0.8



    axs.plot(0, 0,'-' , color=col_1, label ='$true$')
    axs.plot(0, 0,'-', color=col_2,linewidth=2, label='$pred$')

    axs.set_title(title, fontsize=25, pad= 50)

    if zt.shape[2]==2:
        for p_id in range(zt.shape[1]):
            axs.plot(zt[:, p_id, delay_id[0]].to(cpu).detach().numpy(), zt[:, p_id, delay_id[1]].to(cpu).detach().numpy(),'-' , color=col_1)
            axs.plot(zp[:, p_id, delay_id[0]].to(cpu).detach().numpy(), zp[:, p_id, delay_id[1]].to(cpu).detach().numpy(),'-', color=col_2,linewidth=2)
        

        axs.set_xlim([-1.1,1.1])
        axs.set_ylim([-1.1,1.1])
        axs.xaxis.set_ticks(np.linspace(-1,1,3)) 
        axs.yaxis.set_ticks(np.linspace(-1,1,3)) 
        

        axs.set_xlabel(axis_labels[0],fontsize=25)
        axs.set_ylabel(axis_labels[1],fontsize=25)
        axs.grid()
        
    else:
        for i in range(1):
            for p_id in range(zp.shape[1]):
                axs.plot(zt[:, p_id, delay_id[0] ].to(cpu).detach().numpy(), 
                         zt[:, p_id, delay_id[1] ].to(cpu).detach().numpy(),
                         zt[:, p_id, delay_id[2] ].to(cpu).detach().numpy(), '-', color=col_1)

                axs.plot(zp[:, p_id, delay_id[0] ].to(cpu).detach().numpy(), 
                         zp[:, p_id, delay_id[1] ].to(cpu).detach().numpy(),
                         zp[:, p_id, delay_id[2] ].to(cpu).detach().numpy(),'-', color=col_2, linewidth=1)    

        axs.set_xlim([-1.1,1.1])
        axs.set_ylim([-1.1,1.1])
        axs.set_zlim([-1.1,1.1])
                
        axs.xaxis.set_ticks(np.linspace(-1,1,3)) 
        axs.yaxis.set_ticks(np.linspace(-1,1,3)) 
        axs.zaxis.set_ticks(np.linspace(-1,1,3)) 



        axs.set_xlabel(axis_labels[0],fontsize=25)
        axs.set_ylabel(axis_labels[1],fontsize=25)
        axs.set_zlabel(axis_labels[2],fontsize=25)
        
        axs.set_facecolor('white') 

    axs.tick_params(axis='both', which='major', labelsize=20)
    axs.tick_params(axis='both', which='minor', labelsize=20)

    axs.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc='lower left',
                ncol=2, mode="expand", borderaxespad=0., frameon=False, fontsize = 25)
    

    return axs




def get_var_list(delay, var_names):
    var_list = dict()
    start_ids = []
    count = 0
    for i in range(len(var_names)):
        start_ids.append(count)
        for j in range(len(delay[i])):
            var_list[count]= var_names[i]
            count +=1
    return var_list, start_ids

def get_delay_plot_id(delay_id, tab_sup, var_names):
    delay_id_plot = []
    axis_labels = []
    for i in range(len(delay_id)):
        delay_id_temp = []
        axis_lab_temp = []
        delay = tab_sup.iloc[i].Delay
        var_list, start_idx = get_var_list(delay, var_names)

        for j in range(len(delay_id[i])):
            ii = delay_id[i][j][0]
            jj = delay_id[i][j][1]

            idx = start_idx[ii]+ jj
            delay_id_temp.append(idx)
            axis_lab_temp.append('$'+var_list[idx]+'(t+\\tau_{'+ str(ii+1)+','+str(jj+1) +'})$')

        delay_id_plot.append(delay_id_temp)
        axis_labels.append(axis_lab_temp)
    return delay_id_plot, axis_labels

def plot_attractor(fig, n_row, n_col, data_files, param_df, idx_arr, case_names, var_names, color_true, color_pred, delay_id, iter_plot, tab_sup):
    axs = []
    kk = 1
    alpha = ord('a')
    delay_id_plot, axis_labels = get_delay_plot_id(delay_id, tab_sup, var_names)
    n_color = len(color_true)

    for i in range(len(idx_arr)):
        print('kk=',kk)
        f = data_files[i]
        idx = idx_arr[f]

        folder = param_df.loc[idx].iloc[i].folder
        print(folder)
        img_pkl = './'+folder+'/comp_pred_'+str(iter_plot)+'.pkl'
        

        if os.path.exists(img_pkl):
            with open(img_pkl,'rb') as fl:
                img_data = pkl.load(fl)

            zt   =img_data[3]
            print(zt.shape)
            if zt.shape[2]>2:
                axs.append(fig.add_subplot(n_row,n_col,kk, projection='3d'))
                print("3d")
            else:
                axs.append(fig.add_subplot(n_row,n_col,kk))#, projection='2d'))
                print("2d")

            title = '$('+chr(alpha)+')\:'+case_names[f]+'$'

            img_np = plot_cmp_2(axs[kk-1],  title= title, #img_data[0], 
                                            kk   = img_data[1], 
                                            tt   = img_data[2],
                                            zt   = img_data[3], 
                                            tp   = img_data[4],
                                            zp   = img_data[5], 
                                            τ    = img_data[6],
                                            col_1= color_true[(kk-1)%n_color],
                                            col_2= color_pred[(kk-1)%n_color],
                                            delay_id    = delay_id_plot[kk-1],
                                            axis_labels = axis_labels[kk-1])  
            kk +=1
            alpha +=1
    fig.tight_layout(pad=2)


def get_loss_evolution_in_time(param_df, tab_sup, idx_arr, data_files_lyapunov, T_ly, var_names, tab_rows, batch_time_mul):
    z_true_batch_out = dict.fromkeys(data_files_lyapunov)
    z_pred_batch_out = dict.fromkeys(data_files_lyapunov)
    
    loss_batch_out = dict.fromkeys(data_files_lyapunov)
    loss_batch_avg_out = dict.fromkeys(data_files_lyapunov)

    t_batch_out = dict.fromkeys(data_files_lyapunov)

    diameter = dict.fromkeys(data_files_lyapunov)
    for case_idx in tab_rows:
        f = data_files_lyapunov[case_idx]


        z_batch_var = dict.fromkeys(var_names) 
        z_pred_var  = dict.fromkeys(var_names) 

        case_loss = dict.fromkeys(var_names) 
        case_loss_avg = dict.fromkeys(var_names) 
        diameter_temp = dict.fromkeys(var_names) 

        case_details = param_df.loc[idx_arr[f]].iloc[case_idx]
        tab_details  = tab_sup.iloc[case_idx]

        folder = case_details.folder
        print('folder: ', folder)

        data_file = case_details.datafile
        print('data_file: ', data_file)

        τ_arr = tab_details.Delay
        Nc = case_details.Nc
        batch_time = case_details.batch_time
        batch_size = case_details.batch_size


        func = set_neuralODE_from_existing_weight(folder, 
                                                  dim = tab_details.Dimension,
                                                  n_nodes = case_details.n_nodes, 
                                                  n_layers = case_details.n_layers)

        cpu = torch.device('cpu')
        dtype = torch.float32
        device = cpu

        data = np.loadtxt(data_file)

        t_true = torch.tensor(data[case_details.start_id:case_details.end_id + batch_time*batch_time_mul + Nc,0])
        x_data = torch.tensor(data[case_details.start_id:case_details.end_id + batch_time*batch_time_mul + Nc,1:])

        # # Normalized time series
        x_data_sam = x_data
        x_data_sam = x_data_sam- x_data_sam.mean(0)
        x_data_sam = x_data_sam/x_data_sam.abs().max(0).values

        x_true_sam = (x_data_sam[:,0::]).type(dtype).to(device)
        t_true_sam = (t_true-t_true[0]).type(dtype).to(device)


        # # Time step
        t_true = t_true_sam
        dt = t_true[1] - t_true[0]

        ts_ids = json.loads(case_details.timeseries_index)
        x_true = []
        for ts_id in ts_ids:
            x_true.append(x_true_sam[:,ts_id])

        dt = dt.to(device)

        for i in range(len(var_names)):
            diameter_temp[var_names[i]] = max(x_true[i]) - min(x_true[i])
        
        t_batch, z_batch = mapsr_utils.get_batch(t_true, x_true, Nc, τ_arr, batch_time*batch_time_mul, batch_size, device=device)
        z_pred = odeint(func, z_batch[0,:,:].reshape(z_batch.shape[1], z_batch.shape[2]), t_batch[:,0], options={'dtype':dtype}).to(device)


        var_list, start_ids = get_var_list(tab_details.Delay, var_names)

        for i in range(len(var_names)):
            z_batch_var[var_names[i]]   = z_batch[:,:,start_ids[i]]
            z_pred_var[var_names[i]]    = z_pred[:,:,start_ids[i]]

            case_loss[var_names[i]]     = torch.abs(z_pred[:,:,start_ids[i]]-z_batch[:,:,start_ids[i]]).detach().numpy()
            case_loss_avg[var_names[i]] = np.mean(case_loss[var_names[i]],axis=1)

        t_plot = (t_batch[:,0]-t_batch[0,0])/T_ly[case_idx]


        z_true_batch_out[f] = z_batch_var
        z_pred_batch_out[f] = z_pred_var
        
        loss_batch_out[f] = case_loss
        
        loss_batch_avg_out[f] = case_loss_avg
        t_batch_out[f] = t_plot.detach().numpy()

        diameter[f] = diameter_temp

    return z_true_batch_out, z_pred_batch_out, loss_batch_out, loss_batch_avg_out, t_batch_out, diameter

def set_neuralODE_from_existing_weight(folder, dim, n_nodes, n_layers):
    # sys.path.append(folder)
    # from neuralODE import ODEFunc

    # func = ODEFunc(dimensions=dim)

    func = ffnn.ODEFunc(dimensions      = dim, 
                        n_nodes_hidden  = n_nodes, 
                        n_layers_hidden = n_layers)


    func.load_state_dict(torch.load(folder + '/Weight.pkl'))
    return func

def plot_loss_evolution(figsize, case_names, data_files_in, z_true_in, z_pred_in, loss_in, t_in, id_in, color, output_fld):
    for i in range(len(data_files_in)):
        f = data_files_in[i]
        var_names = list(z_true_in[f].keys())
        n_var = len(var_names)

        t_plot = t_in[f]

        fig = plt.figure(figsize=figsize)

        for j in range(n_var):

            z_true = z_true_in[f][var_names[j]]
            z_pred = z_pred_in[f][var_names[j]]

            loss = loss_in[f][var_names[j]]

            axs = fig.add_subplot(2,n_var,j+1)
            axs.plot(t_plot, z_true[:,id_in].detach().numpy(), '-k', label= '$true$')
            axs.plot(t_plot, z_pred[:,id_in].detach().numpy() ,'--r' ,label='$pred$') 
            axs.set_xlabel("$t/T_\\lambda$", fontsize=25)
            axs.set_ylabel("$"+var_names[j]+"$", fontsize=25)
            axs.set_xlim(min(t_plot),max(t_plot))


            axs = fig.add_subplot(2,n_var,n_var+j+1)
            axs.plot(t_plot,loss[:,id_in], color=color[i%len(color)])
            axs.set_xlabel("$t/T_\\lambda$", fontsize=25)
            axs.set_ylabel("$|"+var_names[j]+ "_{pred}-"+var_names[j]+ "_{true}|$", fontsize=25)
            axs.set_xlim([min(t_plot),max(t_plot)])

            axs.legend(loc='upper right')

        fig.suptitle("$"+case_names[f]+"$",fontsize = 30)
        fig.tight_layout()
        fig.savefig(output_fld + '/'+ case_names[f]+"_pred_single.pdf")

        # fig = plt.figure()
        # axs = fig.add_subplot(1,1,1,projection='3d')
        # post_process_mapsr.plot_cmp_2(axs, title= '$'+ppm.case_names[f]+'$', 
        #                                     kk   = 1, 
        #                                     tt   = t_plot,
        #                                     zt   = z_true, 
        #                                     tp   = t_plot,
        #                                     zp   = z_pred, 
        #                                     τ    = τ_arr,
        #                                     col_1= 'brown',
        #                                     col_2= 'darkorange',
        #                                     delay_id    = [0,1,2],
        #                                     axis_labels = ['$x_1$','$x_2$','$x_3$'])  


def get_selective_case_names(dict_in, keys):
    d = dict()
    for key in keys:
        d[key] = dict_in[key]
    return d

def plot_loss_evolution_average(fig, n_row, n_col, case_names, data_files_in, 
                                loss_avg_in, t_in, diameter, color, ylim, output_fld):
    alpha = 'a'
    
    for i in range(len(data_files_in)):
        f = data_files_in[i]
        var_names = list(loss_avg_in[f].keys())
        n_var = len(var_names)

        t_plot = t_in[f]
        axs = fig.add_subplot(n_row,n_col,i+1)

        for j in range(n_var):

            delta = loss_avg_in[f][var_names[j]]
            dia  = diameter[f][var_names[j]]

            
            axs.plot(t_plot,delta/dia, color=color[j], label = "$"+var_names[j]+"$")

        axs.set_xlabel("$t/T_\\lambda$", fontsize=25)
        axs.set_ylabel("$\\bar{\delta}/\delta_{max}$", fontsize=25)

        axs.set_xlim([min(t_plot)-0.01*(max(t_plot)-min(t_plot)),max(t_plot)])
        #axs.set_xlim([-1,max(t_plot)])
        
        axs.set_ylim(ylim)

        #axs.xaxis.set_major_locator(tck.MultipleLocator(5)) 
        axs.yaxis.set_major_locator(tck.MultipleLocator(0.1)) 

        axs.tick_params(axis='both', which='major', labelsize=20)
        axs.tick_params(axis='both', which='minor', labelsize=20)


        axs.set_title("$("+alpha+")\:"+case_names[f]+"$",fontsize = 25)
        axs.grid()
        alpha = chr(ord(alpha)+1)
        axs.legend(fontsize=25)
    plt.tight_layout(pad=5)
    fig.savefig(output_fld + '/'+case_names[f]+"_pred.pdf")