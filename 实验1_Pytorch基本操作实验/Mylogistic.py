import torch
import numpy as np

import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

def sigmod(z):
    return 1 / (1 + torch.exp(-z))

def model(x,w):
    return sigmod(torch.mm(x,w).squeeze(1))

# w1*x1 + w2*x2 = 0 
def x2(x1):
    return -w[0].item() * x1 / w[1].item() 
# 定义二分类交叉熵损失函数
def BCELoss(preds, label):
    return -torch.sum(label * torch.log(preds) + (1 - label) * torch.log(1 - preds)) / len(preds)

if __name__ == "__main__":
    n_data = torch.ones(500, 2)  # 数据的基本形态
    # 从二维高斯分布中取值，有两个均值，分别为[2,2]
    x1 = torch.normal(2 * n_data, 1)
    y1 = torch.zeros(500)  # 类型0
    x2 = torch.normal(-2 * n_data, 1)  # 类型1
    y2 = torch.ones(500)  # 类型1 shape=(500, 1)
    # 注意 x, y 数据的数据形式一定要像下面一样 (torch.cat 是合并数据)
    x = torch.cat((x1, x2), 0).type(torch.FloatTensor)
    y = torch.cat((y1, y2), 0).type(torch.FloatTensor)

    # plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=y.data.numpy(), s=100, lw=0, cmap='RdYlGn')
    # plt.show()
    X_train, X_test, y_train, y_test = train_test_split(x, y, shuffle=True)

    w = torch.tensor(np.random.normal(0, 1, (2, 1)), dtype=torch.float32)
    w.requires_grad_(requires_grad=True)
    epochs = 500
    lr = 0.1
    losses = []
    for i in range(epochs):
        preds = model(X_train, w)
        loss = BCELoss(preds, y_train)
        losses.append(loss.item())
        # 计算梯度
        loss.backward()
        # 更新参数w
        w.data -= lr * w.grad
        # 清空梯度
        w.grad.data.zero_()
        print("epoch {}, erro : {}".format(i, loss))

    test = model(X_test, w)
    test = test.detach()
    test[test >= 0.5] = 1
    test[test < 0.5] = 0
    print("accuracy：")
    print(torch.sum(test == y_test).item() / len(y_test))

    x_label = np.arange(1,501)
    plt.plot(x_label,losses)


    x_plot = np.linspace(-4,4,1000)
    y_plot = x2(x_plot)
    
    plt.plot(x_plot,y_plot)
    plt.scatter(x.data.numpy()[:, 0], x.data.numpy()[:, 1], c=y.data.numpy(), s=100, lw=0, cmap='RdYlGn')
    plt.show()