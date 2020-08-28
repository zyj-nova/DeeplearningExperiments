from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons
import torch
import torch.nn as nn
import numpy as np
from matplotlib import pyplot as plt

def relu(x):
    return torch.where(x > 0, x , torch.tensor(0.0))
    
def network(x):
    H = torch.mm(x,W1) + b1
    H = relu(H)
    
    return torch.sigmoid(torch.mm(H,W2) + b2)

if __name__ == "__main__":
    # 生成数据集
    x,y = make_moons(500,noise=0.1,shuffle=False)
    plt.scatter(x[:250,0],x[:250,1],label='0')
    plt.scatter(x[250:,0],x[250:,1],label='1')
    plt.legend()
    plt.show()

    x = torch.tensor(x).type(torch.float)
    y = torch.tensor(y).type(torch.float)
    X_train, X_test, y_train, y_test = train_test_split(x, y, shuffle=True)

    W1 = torch.normal(0,0.01,(2,64))
    b1 = torch.zeros(64)
    W2 = torch.normal(0,0.01,(64,1))
    b2 = torch.zeros(1)

    params = [W1,W2,b1,b2]
    for param in params:
        param.requires_grad_(requires_grad = True)
    
    critic = nn.BCELoss()

    lr = 0.2
    train_erro = []
    test_erro = []
    epoch = 200
    for i in range(epoch):

        preds = network(X_train)
        loss = critic(preds, y_train)
        print(loss.item())
        loss.backward()
        for param in params:
            param.data = param - param.grad * lr

        for param in params:
            param.grad.data.zero_()
        train_erro.append(loss.item())
        
        test_preds = network(X_test)
        test_erro.append(critic(test_preds.squeeze(1),y_test).item())
    test_preds = network(X_test).squeeze(1)
    test_preds[test_preds >= 0.5] = 1
    test_preds[test_preds < 0.5] = 0

    print(torch.sum(test_preds == y_test) *1.0/ len(test_preds))

    train_erro = np.array(train_erro)
    test_erro = np.array(test_erro)
    axis = np.linspace(1,epoch + 1,epoch)
    plt.plot(axis,train_erro,color='r',label="train loss")
    plt.plot(axis,test_erro,label="test loss")
    plt.legend()
    plt.show()
    

