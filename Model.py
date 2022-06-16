import torch 
from torch.nn import Linear, ELU, Sequential
from torch import nn 
import numpy as np 

# check if there is a CUDA compatible GPU
device = torch.device("cpu")
if torch.cuda.is_available():
    device = torch.device("cuda")

class DQN(nn.Module):
    def __init__(self, input_shape, output_shape):
        super().__init__()
    
        self.linear = Sequential(
            Linear(input_shape, 32),
            ELU(),
            Linear(32, 32),
            ELU(),
            Linear(32, output_shape)
        )

    def forward(self, x):
        x = x.to(device)
        return self.linear(x)


    





