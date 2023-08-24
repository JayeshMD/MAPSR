%%
clc
close all
clear

mpath = mfilename()

%%
base_fld = '/mnt/storage0/jayesh/Work/Jayesh/Work/0_Experimental_Data/Turbulent_Combustor_Vishnu_JFM/28_SLPM/';

file_id_arr  = [2,8,16];
start_id_arr = [15000, 25000, 20000];
end_id_arr   = [18000, 28000, 23000];

% start_id_arr = [15000, 15000, 15000];
% end_id_arr   = [25000, 25000, 25000];

for i=1:length(file_id_arr)
    path = strcat(base_fld,num2str(file_id_arr(i)),'.txt')
    data = dlmread(path);
    
    [m,n]  = size(data);
    
    figure(1)
    
    start_id = start_id_arr(i);
    end_id   = end_id_arr(i);
    y = data(start_id:end_id,2:3);         
    y = y+rand(length(y),1)*1e-3;
    
    plot(y);
   
    id = i+1;
    [Y_reconstruct{i}, tau_vals{i}, ts_vals{i}, Ls{i}, E{i}] = pecuzal_embedding(y, 0:50, 'theiler', 7, 'econ', true);
end

%%
%%
clc
Case=[{'Chaos'};{'Intermittency'};{'LCO'}];
Method = [];
Dimension = [];
Delay = [];
TimeSeriesID = [];
dt = 1e-4;%data(2,1)- data(1,1);
for i=1:3
    %Case = [Case;strcat("Noise level: ", num2str(i-1),"%")];
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

writetable(tab,'PECUZAL_exp_multi_3000.csv','Delimiter',',')  

fprintf('Done.')