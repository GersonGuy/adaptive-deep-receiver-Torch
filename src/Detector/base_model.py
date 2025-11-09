import torch
import torch.nn as nn


class base_model_generator(nn.Module):
    """ Define generic Base model class for neural network architectures."""
    def __init__(self,input_size,hidden_dim,output_size):

        super().__init__()
        self.fc1 = nn.Linear(input_size,32,bias = True)
        #self.act1 = nn.SELU()
        self.fc2 = nn.Linear(32, hidden_dim)
        self.act2 = nn.SELU()
        self.fc3 = nn.Linear(hidden_dim, output_size)

    def forward(self, x):
        x = (self.fc1(x))
        x = self.act2((self.fc2(x)))
        x = self.fc3(x)
        return x

