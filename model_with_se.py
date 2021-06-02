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
    def __init__(self, files, file2):
        datas=np.zeros((0,24))
        labels=np.zeros((0,1),dtype=np.int64)
        for file1 in files:
            data=loadmat(file1)
            data=data['P_abnormal']
            q=pd.read_excel(file2,header=None)
            q=q.values
        
            data=data[:,[447, 314, 276, 327, 153, 111, 230,   0, 422,  79, 351,  75, 288, 410, 355, 365, 396, 160, 131, 456, 260,  34, 209,  55]]
            datas=np.concatenate((datas,data),axis=0)
            label=np.zeros((data.shape[0],1),dtype=np.int64)
            for j,i in enumerate(range(0,data.shape[0],24)):
                label[i:(i+24),0]=q[j,0]-1
            labels=np.concatenate((labels,label),axis=0)
        model=MinMaxScaler()
        datas=model.fit_transform(datas)
        self.datas = datas.astype(np.float32)
        self.labels=labels


    def __getitem__(self, index):
        
        return self.datas[index, :], self.labels[index, 0]

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

    train_size = 0.8
    BATCH_SIZE = 320
    EPOCH = 100
    LR = 0.001
    files=[
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
    Data = mydata(files, r'.\数据\24分区.xlsx')
    train_size = int(len(Data) * train_size)
    test_size = len(Data) - train_size
    train_data, test_data = random_split(Data, [train_size, test_size])
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE)

    model = Fullnet(24, 24).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    F_loss = torch.nn.CrossEntropyLoss()
    
    train_accuracy=[]
    test_accuracy=[]
    for epoch in range(EPOCH):
        print(f'epoch:{epoch}')
        for step, [x, y] in enumerate(train_loader):
            x, y = x.to(device), y.to(device)
            py = model(x)
            # print(torch.max(py, 1)[1].shape)
            loss = F_loss(py, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch==67:
            torch.save(model,'model.pkl')
            break
        
        with torch.no_grad():
            model.eval()
            tp=[]
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                py = model(x)
                loss = F_loss(py, y)
                accuracy_top1 = top1(py, y, device)
                tp.append(accuracy_top1)
            print(f'train_acc:{np.mean(tp):.6f}')
            train_accuracy.append(np.mean(tp))
            ep=[]
            for tx, ty in test_loader:
                tx, ty = tx.to(device), ty.to(device)
                pty = model(tx)
                loss = F_loss(pty, ty)
                accur_top1 = top1(pty, ty, device)
                ep.append(accur_top1)
            print(f'test_acc:{np.mean(ep):.6f}')
            test_accuracy.append(np.mean(ep))
            model.train()

    # np.savetxt(r'./train_ma.txt',train_accuracy,fmt='%.4f')
    # np.savetxt(r'./test_ma.txt',test_accuracy,fmt='%.4f')
    
    # train_accuracy=np.loadtxt(r'./train_ma.txt')
    # test_accuracy=np.loadtxt(r'./test_ma.txt')
    
    # with plt.style.context(['science', 'ieee']):
        # plt.plot(train_accuracy,label='train accuracy')
        # plt.plot(test_accuracy,label='test accuracy')
        # plt.xlabel('epoch')    
        # plt.ylabel('$acc_{area}1$')
        # plt.legend(loc=4)        
        # plt.autoscale(tight=True)
        # plt.ylim(0.3,1)
        # plt.title('(d) Training model with optimized hyperparameter')
        # plt.savefig('./acc1.jpg')