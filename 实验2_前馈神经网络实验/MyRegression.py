import torch
import numpy as np

from matplotlib import pyplot as plt

def relu(x):
    return torch.where(x > 0, x, torch.tensor(0.0))


def network(x):
    H = torch.mm(x, W1) + b1
    H = relu(H)
    return torch.mm(H, W2) + b2

# 定义均方误差函数
def mse(preds, y):
    return torch.sum((preds.squeeze(1) - y.squeeze(1)) ** 2) / len(preds)

if __name__ == "__main__":
    x_np = np.linspace(-1, 1, 200, dtype=np.float32)
    x = torch.unsqueeze(torch.from_numpy(x_np), dim=1)
    y = 2.0 * x ** 2 + x + 0.2 * torch.randn(x.size())

    W1 = torch.normal(0, 0.01, (1, 32))
    b1 = torch.zeros(32)
    W2 = torch.normal(0, 0.01, (32, 1))
    b2 = torch.zeros(1)

    params = [W1, W2, b1, b2]
    for param in params:
        param.requires_grad_(requires_grad=True)

    lr = 0.1
    for i in range(200):
        preds = network(x)
        loss = mse(preds, y)

        loss.backward()

        for param in params:
            param.data = param - lr * param.grad

        for param in params:
            param.grad.data.zero_()
    preds = network(x)
    preds = preds.detach().numpy()
    plt.plot(x, preds, color='r')
    plt.scatter(x, y)
    plt.show()