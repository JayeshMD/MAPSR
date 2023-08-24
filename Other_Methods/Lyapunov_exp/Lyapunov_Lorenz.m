clc
clear
close all
lorenz_xyz = readtable('../Data/Lorenz/noisy_0.txt');

t = lorenz_xyz.Var1;
dt = t(2) - t(1);
fs = 1/dt;


data(:,1) = lorenz_xyz.Var2(2001:12001);
data(:,2) = lorenz_xyz.Var3(2001:12001);
data(:,3) = lorenz_xyz.Var4(2001:12001);



%load('lorenzAttractorExampleData.mat','data','fs');
plot3(data(:,1),data(:,2),data(:,3));

%%
xdata = data(:,1);
dim = 3;
[~,lag] = phaseSpaceReconstruction(xdata,[],dim)

eRange = 500;
lyapunovExponent(xdata,fs,lag,dim,'ExpansionRange',eRange)


Kmin = 100;
Kmax = 350;
lyapExp = lyapunovExponent(xdata,fs,lag,dim,'ExpansionRange',[Kmin Kmax])

writematrix(lyapExp,'Lyapunov_Lorenz.txt')