clc
clear
close all

base_fld = '/mnt/storage0/jayesh/Work/Jayesh/Work/0_Experimental_Data/Turbulent_Combustor_Vishnu_JFM/28_SLPM/';

file_id_arr  = [2,8];%,16];

start_id_arr = [15000, 15000, 15000];
end_id_arr   = [25000, 25000, 25000];

dim_arr = [7,7,2];

for i=1:length(file_id_arr)
    path = strcat(base_fld,num2str(file_id_arr(i)),'.txt');

    data_all = dlmread(path);

    data = data_all(:,2:end);
    
    fs = 1e4;
    
   figure(100)
   plot(data(:,1));
   hold on
   plot(data(:,2));
    

    xdata = data(:,2);
    dim = dim_arr(i);
    [~,lag] = phaseSpaceReconstruction(xdata,[],dim)
    
    eRange = 80;
    lyapunovExponent(xdata,fs,lag,dim,'ExpansionRange',eRange)
    
    
    Kmin = 20;
    Kmax = 80;
    lyapExp(i) = lyapunovExponent(xdata,fs,lag,dim,'ExpansionRange',[Kmin Kmax])
end

writematrix(lyapExp,'Lyapunov_Exp.txt')