import numpy as np
import random

import torch
import torch.nn as nn
from torch import Tensor
import torch.optim as optim
from torch.utils.data import Dataset,DataLoader,SubsetRandomSampler,ConcatDataset
import torch.nn.functional as F

from sklearn.metrics import r2_score
from modules import OneToTwo,ConcatDist2D,Symmetrize2D,DilatedResidual2D,Cropping2D,UpperTri,DilatedResidual1D,Final,PearsonR

from torchsummary import summary

class SeqNN(nn.Module):
    def __init__(self,n_channel=4,max_len=128):
        super(SeqNN, self).__init__()



        ##CNN+dilated CNN

        self.conv_block_1 = nn.Sequential(
            nn.Conv1d(in_channels=4,out_channels=96,kernel_size=11,padding=5),
            nn.ReLU(),
            nn.BatchNorm1d(96),
            nn.MaxPool1d(2)
        )


        self.convTow = nn.ModuleList()
        for i in range(2):
            self.convTow.append(nn.Sequential(nn.Conv1d(96,96,5,padding=2),nn.ReLU(),nn.MaxPool1d(2)))

        self.dilations = DilatedResidual1D(filters=96,rate_mult=1.75,repeat=8,dropout=0.4)

        self.conv_block_2 = nn.Sequential(
            nn.Conv1d(in_channels=96,out_channels=96,kernel_size=5,padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        self.conv_block_3 = nn.Sequential(
            nn.Conv1d(in_channels=96,out_channels=64,kernel_size=5,padding=2),
            nn.ReLU(),
        )
        self.OnetoTwo = OneToTwo()
        self.concat_dist_2d = ConcatDist2D()
        self.conv_block_2d = nn.Sequential(
            nn.Conv2d(in_channels=65,out_channels=48,kernel_size=3,padding=1)
        )
        self.conv_block1_2d = nn.Sequential(
            nn.Conv2d(in_channels=48,out_channels=24,kernel_size=3,padding=1)
        )
        self.conv_block2_2d = nn.Sequential(
            nn.Conv2d(in_channels=24,out_channels=12,kernel_size=3,padding=1)
        )
        self.symmetrize_2d = Symmetrize2D()
        self.dilated_residual_2d = DilatedResidual2D(dropout=0.1,in_channels=12,kernel_size=3,rate_mult=1.75,repeat=6)
        self.crop_2d =Cropping2D(32)
        self.uppertri = UpperTri()
        self.final = Final()
    def forward(self,x):
        x = self.conv_block_1(x)
        # print(x.size(),1)
        for net in self.convTow:
            x = net(x)
        x = self.dilations(x)
        # print(x.size(),2)
        x = self.conv_block_2(x)
        x = self.conv_block_2(x)
        x = self.conv_block_2(x)
        x = self.conv_block_2(x)
        x = self.conv_block_2(x)
        x = self.conv_block_2(x)
        x = self.conv_block_2(x)
        x = self.conv_block_2(x)
        x = self.conv_block_3(x)
        x = self.OnetoTwo(x)
        x = self.concat_dist_2d(x)
        x = self.conv_block_2d(x.cuda())
        x = self.conv_block1_2d(x)
        x = self.conv_block2_2d(x)
        # print(x.size(),2)
        x = self.symmetrize_2d(x)
        x = self.dilated_residual_2d(x)
        x = self.crop_2d(x)
        x = self.uppertri(x.cpu())
        x = x.transpose(1,2)
        x = self.final(x.cuda())
        return x.squeeze(0).transpose(0,1)




def from_upptri(inputs):
    seq_len = inputs.shape[2]
    output_dim = inputs.shape[1]

    triu_tup = np.triu_indices(seq_len, 2)
    triu_index = list(triu_tup[0] + seq_len * triu_tup[1])
    unroll_repr = inputs.reshape(-1, 1)
    return torch.index_select(unroll_repr, 0,torch.tensor(triu_index))

def calc_R_R2(y_true, y_pred, num_targets, device='cuda:0'):
    '''
    Handles the Pearson R and R2 calculation
    '''
    product = torch.sum(torch.multiply(y_true, y_pred), dim=1)
    true_sum = torch.sum(y_true, dim=1)
    true_sumsq = torch.sum(torch.square(y_true), dim=1)
    pred_sum = torch.sum(y_pred, dim=1)
    pred_sumsq = torch.sum(torch.square(y_pred), dim=1)
    count = torch.sum(torch.ones(y_true.shape), dim=1).to(device)
    true_mean = torch.divide(true_sum, count)
    true_mean2 = torch.square(true_mean)

    pred_mean = torch.divide(pred_sum, count)
    pred_mean2 = torch.square(pred_mean)

    term1 = product
    term2 = -torch.multiply(true_mean, pred_sum)
    term3 = -torch.multiply(pred_mean, true_sum)
    term4 = torch.multiply(count, torch.multiply(true_mean, pred_mean))
    covariance = term1 + term2 + term3 + term4

    true_var = true_sumsq - torch.multiply(count, true_mean2)
    pred_var = pred_sumsq - torch.multiply(count, pred_mean2)
    pred_var = torch.where(torch.greater(pred_var, 1e-12), pred_var, np.inf*torch.ones(pred_var.shape).to(device))

    tp_var = torch.multiply(torch.sqrt(true_var), torch.sqrt(pred_var))

    correlation = torch.divide(covariance, tp_var)
    correlation = correlation[~torch.isnan(correlation)]
    correlation_mean = torch.mean(correlation)
    total = torch.subtract(true_sumsq, torch.multiply(count, true_mean2))
    resid1 = pred_sumsq
    resid2 = -2*product
    resid3 = true_sumsq
    resid = resid1 + resid2 + resid3
    r2 = torch.ones_like(torch.tensor(num_targets)) - torch.divide(resid, total)
    r2 = r2[~torch.isinf(r2)]
    r2_mean = torch.mean(r2)
    return correlation_mean, r2_mean



import h5py
class MyDataset(Dataset):
    def __init__(self,data_path):
        self.data = h5py.File(data_path,"r")["data"]
        self.label = h5py.File(data_path, "r")["label"]
    def __len__(self):
        return len(self.data)
    def __getitem__(self, index):
        return self.data[index],self.label[index]

dataset = MyDataset("/home/sun/data1/work/deep_learning/data/train.h5")

dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = SeqNN()
model.load_state_dict(torch.load("train1.h5"))

# print(summary(model.cuda(),(4,524288)))##print model

loss_fn = nn.MSELoss()
optimzer = optim.Adam(model.parameters(),lr=0.0001)

model.to(device)



# pear = PearsonR(1)

for j in range(60):
    for index,(data,label) in enumerate(dataloader):
        y_pred = model(data.transpose(1,2).to(torch.float32).to(device))
        label = from_upptri(label).transpose(0,1).to(torch.float32)
        # label = F.normalize(label, p=2, dim=-1,eps = 1e-6)
        # print(y_pred)
        # print(label)
        # print(y_pred.size(),label.size())
        # print(y_pred.size(),label.transpose(0,1))
        loss = loss_fn(y_pred,label.cuda())
        # r = pear(label.cuda(),y_pred)
        r,r2 = calc_R_R2(y_true=label.cuda(),y_pred=y_pred,num_targets=1)
        # for i in range(label.shape[1]):
            # print(y_pred[0][i],label[0][i])
            # loss_ = loss_fn(y_pred[0][i],label[0][i].cuda())
            # loss += loss_

        optimzer.zero_grad()
        loss.backward()
        optimzer.step()

        if (index+1) % 5 == 0:
            print("\r[{0}/{1}]===>loss:{2}===> persor:{3},r2:{4}".format(j,60,loss,r,r2))


###eval
