from scipy.io import loadmat
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.tensorboard import SummaryWriter
from torchsummary import summary
import time
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import classification_report,confusion_matrix
import sys
from math import ceil
import optuna
import matplotlib.pyplot as plt
# plt.style.use(['science','ieee'])

torch.manual_seed(88)
torch.cuda.manual_seed(88)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.manual_seed(88)
torch.cuda.manual_seed(88)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class block(nn.Module):
    def __init__(self,in_features,hidden1,hidden2,hidden3,out_features):
        super(block,self).__init__()
        self.short=nn.Sequential(
                    nn.BatchNorm1d(in_features),
                    nn.ReLU()
                )
        self.density=nn.Sequential(
                    nn.Linear(in_features,hidden1),
                    nn.ReLU(),
                    nn.Linear(hidden1,hidden2),
                    nn.ReLU(),
                    nn.Linear(hidden2,hidden3),
                    nn.ReLU(),
                    nn.Linear(hidden3,out_features)
                    ,nn.Dropout(0.5)
                )

    def forward(self,x):
        x=self.density(self.short(x))
        return x

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        # self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # b, c, _, _ = x.size()
        # y = self.avg_pool(x).view(b, c)
        y = self.fc(x)
        return x * y.expand_as(x)
        
class Fullnet(nn.Module):
    def __init__(self,in_features,out_features):
        super(Fullnet,self).__init__()
        self.input=nn.Linear(in_features,68)
        self.block1=block(68, 113, 281, 83, 68)
        self.se1=SELayer(68,8)
        self.block2=block(68, 138, 283, 128, 68)
        self.se2=SELayer(68,8)
        self.block3=block(68, 160, 315, 139, 68)
        self.se3=SELayer(68,8)
        self.block4=block(68, 120, 231, 107, 68)
        self.se4=SELayer(68,8)
        self.block5=block(68, 99, 151, 139,out_features)
        self.output=nn.Softmax(dim=1)

    def forward(self,x):
        x=self.input(x)
        d1=x
        x=self.block1(d1)
        x=self.se1(x)
        d2=x+d1
        x=self.block2(d2)
        x=self.se2(x)
        d3=x+d1+d2
        x=self.block3(d3)
        x=self.se3(x)
        d4=x+d1+d2+d3
        x=self.block4(d4)
        x=self.se4(x)
        d5=x+d1+d2+d3+d4
        x=self.block5(d5)
        x=self.output(x)
        return x

class mydata(Dataset):
    def __init__(self,p_files, q_files,file):
        datas=np.zeros((0,24))
        qs=np.zeros((0,1),dtype=np.int64)
        labels=np.zeros((0,1),dtype=np.int64)
        for file1,file2 in zip(p_files, q_files):
            data=loadmat(file1)
            data=data['P_abnormal']
            q=loadmat(file2)
            q=q['Q_burst']
        
            data=data[:,[447, 314, 276, 327, 153, 111, 230,   0, 422,  79, 351,  75, 288, 410, 355, 365, 396, 160, 131, 456, 260,  34, 209,  55]]
            datas=np.concatenate((datas,data),axis=0)
            qq=np.zeros((data.shape[0],1))
            for i in range(q.shape[1]):
                qq[i*24:i*24+24,0]=q[:,i]
            qs=np.concatenate((qs,qq),axis=0)
            la=pd.read_excel(file,header=None)
            la=la.values
            label=np.zeros((data.shape[0],1),dtype=np.int64)
            for j,i in enumerate(range(0,data.shape[0],24)):
                label[i:(i+24),0]=la[j,0]-1
            labels=np.concatenate((labels,label),axis=0)
        model=MinMaxScaler()
        datas=model.fit_transform(datas)
        self.datas = datas.astype(np.float32)
        self.qs=qs
        self.labels=labels


    def __getitem__(self, index):
        
        return self.datas[index, :], self.qs[index, 0],self.labels[index, 0]

    def __len__(self):
        return self.datas.shape[0]


def top5(precision, true, device):
    value, index = torch.topk(precision, 5, dim=1)
    numbers = true.shape[0]
    accuracy = torch.zeros(numbers).to(device)
    for i in range(numbers):
        if true[i] in index[i, :]:
            accuracy[i] = 1
    return (torch.sum(accuracy) / torch.Tensor([numbers]).to(device)).item()


def top1(precision, true, device):
    index = torch.max(precision, 1)[1]
    accuracy = sum(index == true) / torch.Tensor([true.shape[0]]).to(device)
    return accuracy.item()



if __name__ == "__main__":

    p_files=[
    ".\数据\9月4日高日_P_0.2.mat",
    ".\数据\9月4日高日_P_0.4.mat",
    ".\数据\9月4日高日_P_0.5.mat",
    ".\数据\9月4日高日_P_0.6.mat",
    ".\数据\9月4日高日_P_0.22.mat",
    ".\数据\9月4日高日_P_0.24.mat",
    ".\数据\9月4日高日_P_0.26.mat",
    ".\数据\9月4日高日_P_0.28.mat",
    ".\数据\9月4日高日_P_0.32.mat",
    ".\数据\9月4日高日_P_0.34.mat",
    ".\数据\9月4日高日_P_0.36.mat",
    ".\数据\9月4日高日_P_0.38.mat",
    ".\数据\9月4日高日_P_0.42.mat",
    ".\数据\9月4日高日_P_0.44.mat",
    ".\数据\9月4日高日_P_0.46.mat",
    ".\数据\9月4日高日_P_0.48.mat",
    ".\数据\9月4日高日_P_0.52.mat",
    ".\数据\9月4日高日_P_0.54.mat",
    ".\数据\9月4日高日_P_0.56.mat",
    ".\数据\9月4日高日_P_0.58.mat"
    ]
    q_files=[
    ".\数据\9月4日高日_Q_0.2.mat",
    ".\数据\9月4日高日_Q_0.4.mat",
    ".\数据\9月4日高日_Q_0.5.mat",
    ".\数据\9月4日高日_Q_0.6.mat",
    ".\数据\9月4日高日_Q_0.22.mat",
    ".\数据\9月4日高日_Q_0.24.mat",
    ".\数据\9月4日高日_Q_0.26.mat",
    ".\数据\9月4日高日_Q_0.28.mat",
    ".\数据\9月4日高日_Q_0.32.mat",
    ".\数据\9月4日高日_Q_0.34.mat",
    ".\数据\9月4日高日_Q_0.36.mat",
    ".\数据\9月4日高日_Q_0.38.mat",
    ".\数据\9月4日高日_Q_0.42.mat",
    ".\数据\9月4日高日_Q_0.44.mat",
    ".\数据\9月4日高日_Q_0.46.mat",
    ".\数据\9月4日高日_Q_0.48.mat",
    ".\数据\9月4日高日_Q_0.52.mat",
    ".\数据\9月4日高日_Q_0.54.mat",
    ".\数据\9月4日高日_Q_0.56.mat",
    ".\数据\9月4日高日_Q_0.58.mat"
    ]
    batch_size = 464*24
    index = torch.zeros((batch_size,1))
    for i,j in enumerate(range(0,464*24,24)):
        index[j:j+24] = i+1
    Data = mydata(p_files,q_files,r'.\数据\24分区.xlsx')
    Data_loader = DataLoader(Data, batch_size=batch_size, shuffle=False)

    model = torch.load('model.pkl').to('cpu')
    model.eval()
    
    analyse=torch.zeros((len(Data),27))
    
    print('开始分析流量：->')
    for num,(p,q,l) in enumerate(Data_loader):
        py = model(p)
        pl=torch.argsort(py,1,descending=True)
        analyse[num*batch_size:num*batch_size+batch_size,1] = q
        analyse[num*batch_size:num*batch_size+batch_size,2] = l
        analyse[num*batch_size:num*batch_size+batch_size,3:] = pl
        analyse[num*batch_size:num*batch_size+batch_size,0] = index[:,0]
    col = ['pre_top%s'%i for i in range(1,25)]
    col.insert(0,'true_area_id')
    col.insert(0,'flow')
    analyse=pd.DataFrame(analyse.numpy()[:,1:],columns=col,index=analyse.numpy()[:,0])
    analyse.to_csv('流量分析.csv')
    print('分析完成！')