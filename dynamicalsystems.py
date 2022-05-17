import torch 

def Lorenz(xx,p):
    x, y, z = xx[0], xx[1], xx[2]
    σ, ρ, β =  p[0],  p[1],  p[2]
    return torch.tensor([ σ*(y-x), x*(ρ-z)-y, x*y-β*z])

def DampedOscillator(xx,p):
    x, y = xx[0], xx[1]
    ω, ζ = p[0], p[1]
    return torch.tensor([ y, -ω**2*x - ζ*y])