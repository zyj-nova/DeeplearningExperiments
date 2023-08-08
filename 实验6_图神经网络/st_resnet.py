# %%
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict
from gcn import GCN
# %%
class ResUnit(nn.Module):
    def __init__(self,in_dim,nb_filter=64):
        super(ResUnit,self).__init__()
        self.bn1 = nn.BatchNorm2d(in_dim)
        self.conv1 = nn.Conv2d(in_dim,nb_filter,3,1,1)
        self.bn2 = nn.BatchNorm2d(in_dim)
        self.conv2 = nn.Conv2d(in_dim,nb_filter,3,1,1)
    def forward(self,x):
        out = self.bn1(x)
        out = F.relu(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = F.relu(out)
        out = self.conv2(out)
        out += x
        return out

    
class ResSeq(nn.Module):
    def __init__(self,in_dim,nb_filter,nb_residual_unit):
        super(ResSeq,self).__init__()
        layers = []
        for i in range(nb_residual_unit):
            layers.append(ResUnit(in_dim,nb_filter))
        self.resLayer = nn.Sequential(*layers)
    def forward(self,x):
        return self.resLayer(x)
        
class ResConvUnits(nn.Module):
    def __init__(self,in_dim, out_dim, nb_residual_unit,nb_filter=64):
        super(ResConvUnits,self).__init__()
        self.conv1 = nn.Conv2d(in_dim,64,3,1,1)
        self.resseq = ResSeq(64,nb_filter,nb_residual_unit)
        self.conv2 = nn.Conv2d(nb_filter,out_dim,3,1,1)
    def forward(self,x):
        
        out = self.conv1(x)
        out = self.resseq(out)
        out = F.relu(out)
        return self.conv2(out)
        
class External(nn.Module):
    def __init__(self,external_dim):
        super(External,self).__init__()
        self.embed = nn.Linear(external_dim,10)
        self.fc = nn.Linear(10,2)
    def forward(self,x):
        x = x.permute(0,2,3,1)
        #print("x",x.shape)
        out = self.embed(x)
        
        out = F.relu(out)
        out = self.fc(out)
    
        out = F.relu(out)
        return out