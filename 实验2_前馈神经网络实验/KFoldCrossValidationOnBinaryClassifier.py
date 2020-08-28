from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def get_k_fold_data(k, i, X, y):  ###此过程主要是步骤（1）
    # 返回第i折交叉验证时所需要的训练和验证数据，分开放，X_train为训练数据，X_valid为验证数据
    assert k > 1
    fold_size = X.shape[0] // k  # 每份的个数:数据总条数/折数（组数）
    
    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)  #slice(start,end,step)切片函数
        ##idx 为每组 valid
        X_part, y_part = X[idx, :], y[idx]
        if j == i: ###第i折作valid
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = torch.cat((X_train, X_part), dim=0) #dim=0增加行数，竖着连接
            y_train = torch.cat((y_train, y_part), dim=0)
    #print(X_train.size(),X_valid.size())
    return X_train, y_train, X_valid,y_valid

class BinaryClassifer(nn.Module):
    def __init__(self):
        super().__init__()
        self.hid1 = nn.Linear(2,64)
        self.hid2 = nn.Linear(64,32)
        self.out = nn.Linear(32,1)
        
    def forward(self,x):
        h = self.hid1(x)
        h = F.relu(h)
        
        h = self.hid2(h)
        h = F.relu(h)
        
        o = self.out(h)
        o = torch.sigmoid(o)
        return o

if __name__ == "__main__":
    x,y = make_moons(500,noise=0.1,shuffle=False)
    x = torch.tensor(x).type(torch.float)
    y = torch.tensor(y).type(torch.float)

    X_train, X_test, y_train, y_test = train_test_split(x, y, shuffle=True)

    k = 10
    epochs = 100
    # k折交叉验证
    for i in range(k):
        train_loss_sum, valid_loss_sum = 0, 0
        
        model = BinaryClassifer()
        lr = 0.2
        criteria = nn.BCELoss()
        optimizer = optim.SGD(model.parameters(),lr =lr)
        for epoch in range(epochs):
            Xtrain,ytrain, X_valid, y_valid = get_k_fold_data(10,i,X_train,y_train)
            preds = model(Xtrain)
            loss = criteria(preds.squeeze(1),ytrain)
            train_loss_sum += loss.item()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            valid_preds = model(X_valid)
            valid_loss = criteria(valid_preds.squeeze(1),y_valid)
            valid_loss_sum += valid_loss.item()
        print("#{}折：train_loss : {}, valid_loss : {}".format(i,train_loss_sum, valid_loss_sum))
    
    y_hat = model(X_test)
    y_hat[y_hat >= 0.5] = 1
    y_hat[y_hat < 0.5] = 0
    print(torch.sum(y_hat.squeeze(1) == y_test).item() / len(y_test))