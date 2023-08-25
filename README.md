# MAPSR method

We introduced the model adaptive phase space reconstruction (MAPSR) method. A brief description of the method is given below.

Suppose we have a time series of $x(t)$. We start with an array of delays $\tau=[\tau_1, \tau_2 ,...,\tau_n]$. Here, $\tau_i<\tau_j$ if $ i < j $, i.e. 
delays are arranged in ascending order and $\tau_i \geq 0$. The $x$ corresponding to the delay state is obtained using interpolation. 

We demonstrate the MAPSR method with ODE ($f$) modeled using polynomial and neural ODE. 

$$\dot{z}=f(z)$$

Here, $z(t)=[x(t+\tau_1), x(t+\tau_2), x(t+\tau_3)..., x(t+\tau_n)]$ is the delay vector.

 The ODE model is integrated with the initial condition $z_0$. The loss function $\mathcal{L}$ decides the quality of fit as well the suitability of reconstruction to the model. The sensitivity of $\mathcal{L}$ to model parameters and delay vector is obtained using the _back propation_ method from _PyTorch_. The gradient descent method is used to update model parameters and delay vectors. 

 If consecutive delays are closer than the specified accuracy/tolerance, the values are merged, or one of the delays is removed from the array. The training is then again restarted.

 # Follow the following steps to apply the MAPSR method to the time series.

  1. Create a folder with the Case name containing a <em>Param.csv</em> file. As an example,
     1. HarmonicOscillator
     2. Lorenz
     3. Exp_Tara
     4. Exp_Tara_multivariate,are already created with the 'Param.csv' file.
2. Set the <em>param_df_in</em> variable to path of <em>Param.csv</em> file in <em>run_case.py</em>.
 
 To run the code say with 20 processors,
      
    mpirun -n 20 python run_case.py
 
 or
   
    mpiexec -n 20 python run_case.py

# Postprocessing

To analyze the results of the above test cases, use the following files, which give an overview of the available tools.

1. analyze_HarmonicOscillator.ipynb
2. analyze_Lorenz.ipynb
3. analyze_Exp_Tara.ipynb
4. analyze_Exp_Tara_multivariate.ipynb
