'''
SimAM：A Simple，Parameter-Free Attention Module for Convolutional Neural Networks
http://proceedings.mlr.press/v139/yang21o/yang21o.pdf
https://github.com/ZjjConan/SimAM
'''

import torch
from torch import nn


class SimAM(torch.nn.Module):
    def __init__(self, e_lambda=1e-4):
        super(SimAM, self).__init__()
        self.activaton = nn.Sigmoid()
        self.e_lambda = e_lambda


    def forward(self, x):
        b, c, h, w = x.size()
        n = w * h - 1
        x_minus_mu_square = (x - x.mean(dim=[2, 3], keepdim=True)).pow(2)
        y = x_minus_mu_square / (4 * (x_minus_mu_square.sum(dim=[2, 3], keepdim=True) / n + self.e_lambda)) + 0.5

        return x * self.activaton(y)
        
    
if __name__ == '__main__':
    img = torch.rand(2, 6, 256, 256)
    b, c, h, w = img.shape
    net = SimAM()
    output = net(img)
    print(img.shape)
    print(output.shape)