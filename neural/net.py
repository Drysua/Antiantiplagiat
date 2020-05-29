import torch
import torch.nn as nn

import pandas as pd
import numpy as np

batch_size    = 100
noise_size    = 296
size = 400
w = 80
dtype         = torch.FloatTensor

def sample_noise(batch_size, dim):
    return (torch.rand([batch_size, dim])*2 - 1).type(dtype)

class Flatten(nn.Module):
    
    def forward(self, x):
        N, C, H, W = x.size()
        return x.view(N, -1)  
    
class Unflatten(nn.Module):
    
    def __init__(self, N=-1, C=128, H=7, W=7):
        super(Unflatten, self).__init__()
        self.N = N
        self.C = C
        self.H = H
        self.W = W
    def forward(self, x):
        return x.view(self.N, self.C, self.H, self.W)

def get_model(PATH):
    
    G = nn.Sequential(
            
            nn.Linear(noise_size,1024),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(1024),
            nn.Linear(1024, 8*w*w),
            nn.ReLU(inplace=True),
            nn.BatchNorm1d(8*w*w),
            Unflatten(batch_size, 128, w // 4, w // 4),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
            
        ).type(dtype)
    
    device = torch.device('cpu')
    G.load_state_dict(torch.load(PATH, map_location=device))
    
    return G