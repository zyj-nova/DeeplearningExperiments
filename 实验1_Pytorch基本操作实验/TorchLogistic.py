import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
# w1*x1 + w2*x2 = 0 
def x2(x1):
    return -w[0][0].item() * x1 / w[0][1].item() 

class LogisticModel(nn.Module):
    def __init__(self,input_dims):
        super().__init__()
        self.linear = nn.Linear(input_dims,1)

    def forward(self,x):
        out = self.linear(x)
        out = 1 / (1 + torch.exp(-out)).squeeze(1)
        return out

if __name__ == "__main__":
    n_data = torch.ones(500, 2)  # 数据的基本形态
    # 从二维高斯分布中取值，有两个均值，分别为[2,2]
    x1 = torch.normal(2 * n_data, 1)  #
    y1 = torch.zeros(500)  # 类型0 shape=(50, 1)
    x2 = torch.normal(-2 * n_data, 1)  # 类型1
    y2 = torch.ones(500)  # 类型1
    # 注意 x, y 数据的数据形式一定要像下面一样 (torch.cat 是合并数据)
    x = torch.cat((x1, x2), 0).type(torch.FloatTensor)
    y = torch.cat((y1, y2), 0).type(torch.FloatTensor)

    # plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=y.data.numpy(), s=100, lw=0, cmap='RdYlGn')
    # plt.show()
    X_train, X_test, y_train, y_test = train_test_split(x, y, shuffle=True)
    lr = 0.1
    epochs = 500

    model = LogisticModel(2)
    loss = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(),lr = lr)
    losses = []
    for i in range(epochs):
        preds = model(X_train)
        e = loss(preds,y_train)
        losses.append(e.item())
        optimizer.zero_grad()
        e.backward()
        optimizer.step()
        print("loss is " + str(e.item()))

    test = model(X_test)
    test = test.detach()
    test[test >= 0.5] = 1
    test[test < 0.5] = 0
    print("accuracy：")
    print(torch.sum(test == y_test).item() / len(y_test))

    x_label = np.arange(1,501)
    plt.plot(x_label,losses)
    w = next(model.parameters()).data
    w.reshape(-1)
    
    x_plot = np.linspace(-4,4,1000)
    y_plot = x2(x_plot)

    plt.plot(x_plot,y_plot)
    plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=y.data.numpy(), s=100, lw=0, cmap='RdYlGn')
    plt.show()