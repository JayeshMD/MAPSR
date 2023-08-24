import torch.nn as nn
class ODEFunc(nn.Module):    
    def __init__(self, dimensions, n_nodes_hidden, n_layers_hidden):
        super(ODEFunc, self).__init__()
        
        modules = []
        
        if n_layers_hidden == 0:
            modules.append(nn.Linear(dimensions, dimensions))
        else:
            modules.append(nn.Linear(dimensions, n_nodes_hidden))
            modules.append(nn.Tanh())
            for i in range(n_layers_hidden-1):
                modules.append(nn.Linear(n_nodes_hidden, n_nodes_hidden))
                modules.append(nn.Tanh())
            modules.append(nn.Linear(n_nodes_hidden, dimensions))
        self.net = nn.Sequential(*modules)
                
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.001)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, y):
        return self.net(y) 