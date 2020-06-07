import torch
import torch.nn as nn

import numpy as np

import os

epoch         = 10000
batch_size    = 5
noise_size    = 296
lr_gen        = 1e-3
lr_dis        = 1e-3
verbose       = 1
dtype         = torch.FloatTensor
w             = 64
n_critic      = 5

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


class Pyramid_G(nn.Module):
    def deconv_block(self, main, name, in_chan, out_chan, ker_size, stride, pad, bn=True):
        main.add_module(f'{name}-{in_chan} {out_chan} convt', nn.ConvTranspose2d(in_chan, out_chan, ker_size, stride, pad, bias=False))
        if bn:
            main.add_module(f'{name}-{out_chan} batchnorm', nn.BatchNorm2d(out_chan))
        main.add_module(f'{name}-{out_chan} relu', nn.ReLU(inplace=True))

    def __init__(self, img_size, img_chan, sec_chan, n_extra_layers=0):
        super(Pyramid_G, self).__init__()

        main = nn.Sequential()

        img_size_copy = img_size
        chan_size = sec_chan

        while img_size_copy > 4:
            chan_size *= 2
            img_size_copy //= 2

        main.add_module('1 Linear', nn.Linear(noise_size, chan_size*4*4))
        main.add_module('1Relu', nn.ReLU())
        main.add_module('1BatchNorm', nn.BatchNorm1d(chan_size*4*4))
        main.add_module('Unflatten', Unflatten(N=-1, C=chan_size, H=4, W=4))
       
        while img_size > 4:
            self.deconv_block(main, 'pyramid', chan_size, chan_size//2, 4, 2, 1)
            chan_size //= 2
            img_size  //= 2

        for t in range(n_extra_layers):
            self.deconv_block(main, f'extra-{t}', chan_size, chan_size, 3, 1, 1)

        main.add_module(f'final {chan_size}-1 conv', nn.ConvTranspose2d(chan_size, img_chan, 3, 1, 1, bias=False))
        main.add_module(f'final tanh', nn.Tanh())

        self.main = main.type(dtype)

    def sample_noise(self, batch_size, noise_size):
        return (torch.rand([batch_size, noise_size])*2 - 1).type(dtype)

    def forward(self, input):
        return self.main(input)
        
def get_model(PATH):
    
    device = torch.device('cpu')
    model = Pyramid_G(64, 3, 64)
    # 
    # model = torch.load(PATH, map_location=device)
    #model = torch.load(PATH)
    model.load_state_dict(torch.load(PATH, map_location=device))
    
    return model

def micro_noise(batch_size, noise_size, speed):
    return ((torch.rand([batch_size, noise_size])*2 - 1)/float(speed)).type(dtype)