from pecuzal_embedding import pecuzal_embedding
import pandas as pd
import numpy as np

def pecuzal_method(param_file):
    params = pd.read_csv(param_file)
    for i in range(0,len(params)):
        '''
        You might need to adjust this routine to get results. 
        Recommended: Use given matlab code for computing embedding 
        parameters. 
        '''

        datafile = params.iloc[i].datafile
        data = np.loadtxt(datafile)
        t = data[:,0]

        x_temp = data[:,1:]
        exec('ts_idx = np.array('+params.iloc[0].timeseries_index+')')
        x_temp = x_temp[:,ts_idx]
        add_noise = input("Do you want to add noise?(y/n)")

        if add_noise.lower() == 'y':
            noise_level = float(input("Noise level:"))
            noise = np.random.normal(0,1,size=len(x_temp))* noise_level
            x = x_temp + noise
        else:
            x = x_temp

        sampling_time = t[1]-t[0]  

        Y_reconstruct, tau_vals, ts_vals, Ls, eps = pecuzal_embedding(x, taus = range(100), theiler = 7, econ = True)

        tab = pd.concat((tab,
                        pd.DataFrame({ "Case":ppm.case_name[datafile],\
                                        "Method":"PECUZAL",\
                                        "Dimension":len(tau_vals), \
                                        "Delay ("+nf_str+")": [np.array(tau_vals)*dt]})))
        
def create_table(param_file, pecuzal_csv, n_var, time_series_name, norm_fact):

    params = pd.read_csv(param_file)
    exec('ts_idx = np.array('+params.iloc[0].timeseries_index+')')

    tab = pd.read_csv(pecuzal_csv)
    tab_2 = tab.copy(deep=True)
    tab_2 = tab_2.drop('TimeSeriesID', axis=1)

    nf_str = "$\\times10^{"+str(int(np.log10(norm_fact)))+"}$"

    delay_pecuzal_all = []
    for kk in range(len(tab)):
        delay_arr = tab.Delay.iloc[kk]
        time_series_id_pecuzal = tab.TimeSeriesID.iloc[kk]

        delay_arr = np.fromstring(delay_arr[1:-1],sep=',', dtype=np.float32)
        
        time_series_id_pecuzal = np.fromstring(time_series_id_pecuzal[1:-1],sep=',', dtype=int)

        delay_pecuzal = [[] for i in range(n_var)]

        for i in range(len(delay_arr)):
            delay_pecuzal[time_series_id_pecuzal[i]-1].append(delay_arr[i])
            
        delay_tab = ''
        for dd in range(len(delay_pecuzal)):
            delay_round = np.array(delay_pecuzal[dd])
            delay_round = np.round(delay_round*norm_fact,4)

            formated_delay = np.array2string(delay_round, precision=4, separator=', ')

            delay_tab += time_series_name[dd]+" : "+ formated_delay + '\n'

        delay_pecuzal_all.append(delay_tab)
        tab_2.loc[kk,"Delay"] = delay_tab

    tab_2.rename(columns={"Delay": "Delay ("+nf_str+")"}, inplace = True)
    return tab_2
        
