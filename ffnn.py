# Creates feed forward neural network
import os

def write_nn(nn_fld,n_layers,n_nodes='10'):
    if not(os.path.exists(nn_fld)):
        os.makedirs(nn_fld)

    if not(isinstance(n_nodes,str)):
        n_nodes = str(n_nodes)

    file_name = nn_fld+'/neuralODE.py'
    text_file = open(file_name,"w")

    nn_text = '''import torch.nn as nn\nclass ODEFunc(nn.Module):    
    def __init__(self, dimensions):
        super(ODEFunc, self).__init__()

        self.net = nn.Sequential(\n'''

    if n_layers==0:
        nn_text +='\t\t\tnn.Linear(dimensions, dimensions),)\n'
    else:
        nn_text +='\t\t\tnn.Linear(dimensions, '+n_nodes+'),\n'
        nn_text +='\t\t\tnn.Tanh(),\n'
        for i in range(n_layers-1): 
            nn_text +='\t\t\tnn.Linear('+n_nodes+', '+n_nodes+'),\n'
            nn_text +='\t\t\tnn.Tanh(),\n'  
        nn_text +='\t\t\tnn.Linear('+n_nodes+', dimensions),)\n'

    nn_text +='''
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.001)
                nn.init.constant_(m.bias, val=0)

    def forward(self, t, y):
        return self.net(y) 
    '''

    text_file.write(nn_text)
    text_file.close()
    
if __name__ == "__main__":
    write_nn('nn.py',2,'dimensions+20')



