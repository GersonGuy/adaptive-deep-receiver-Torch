import torch
import torch.nn as nn


class base_model_generator(nn.Module):
    """ Define generic Base model class for neural network architectures."""
    def __init__(self,input_size,hidden_dim,output_size):

        super().__init__()
        self.fc1 = nn.Linear(input_size,hidden_dim,bias = True)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_size,bias=True)

    def forward(self, x):
        x = self.act1(self.fc1(x))
        x = self.fc2(x)
        return x

