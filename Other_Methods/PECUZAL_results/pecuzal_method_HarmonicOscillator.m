%%
clc
close all
clear

path = mfilename()

%%
base_fld = '/mnt/storage0/jayesh/Work/Jayesh/Work/0_Experimental_Data/Turbulent_Combustor_Vishnu_JFM/28_SLPM/';

noise_levels = 0:5;

start_id_arr = ones(size(noise_levels))*5000;
end_id_arr   = ones(size(noise_levels))*10000;

for i=1:length(noise_levels)
    path = strcat('../Data/HarmonicOscillator/noisy_',num2str(noise_levels(i)),'.txt');
    data = dlmread(path);
    
    [m,n]  = size(data);
    
    figure(1)
    
    start_id = start_id_arr(i);
    end_id   = end_id_arr(i);
    y = data(start_id:end_id,2);         
    y = y+rand(length(y),1)*1e-3;
    
    plot(y);
   
    id = i+1;
    [Y_reconstruct{i}, tau_vals{i}, ts_vals{i}, Ls{i}, E{i}] = pecuzal_embedding(y, 0:50, 'theiler', 7, 'econ', true);
end

%%
%%
clc
Case = [];
Method = [];
Dimension = [];
Delay = [];
TimeSeriesID = [];
dt = 1e-2;
for i=1:length(noise_levels)
    Case = [Case;{strcat("Noise level: ", num2str(noise_levels(i)),"\%")}];
    Method = [Method;{'PECUZAL'}];
    Dimension = [Dimension;length(tau_vals{i})];
    arr = arr_to_str(tau_vals{i}*dt);

    Delay = [Delay;{arr}];
    TimeSeriesID = [TimeSeriesID; arr_to_str(ts_vals{i})];
end
%                                     "Method":"PECUZAL",\
%                                     "Dimension":len(tau_vals), \
%                                     "Delay": [np.array(tau_vals)*dt]}))
tab = table(Case,Method,Dimension,Delay,TimeSeriesID);

writetable(tab,'PECUZAL_HarmonicOscillator.csv','Delimiter',',')  

fprintf('Done.')