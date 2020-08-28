import torch
import torch.nn as nn
from utils import load_data
import time
class MyConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

    def forward(self, X):
        K = [nn.Parameter(torch.normal(0,0.01,(self.in_channels, self.kernel_size, self.kernel_size)), requires_grad=True) for i in
             range(self.out_channels)]
        return self.conv2d_multi_out(X, K)

    def conv2d(self, X, k):
        # 自定义实现单通道卷积
        # X [batch, channel, H, W]
        # K kernel
        batch_size, H, W = X.shape
        k_h, k_w = k.shape
        Y = torch.zeros((batch_size, H - k_h + 1, W - k_w + 1))
        for i in range(Y.shape[1]):
            for j in range(Y.shape[2]):
                for p in range(Y.shape[0]):
                    Y[p, i, j] = (X[p, i: i + k_h, j: j + k_w] * k).sum()
        return Y

    def conv2d_multi_in(self, X, k):
        res = self.conv2d(X[:, 0, :, :], k[0, :, :])
        for i in range(1, X.shape[1]):
            res += self.conv2d(X[:, i, :, :], k[i, :, :])
        return res

    def conv2d_multi_out(self, X, K):
        return torch.stack([self.conv2d_multi_in(X, k) for k in K], dim=1)


class MyConvNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = MyConv2d(3, 1, 11)
        self.fc = nn.Linear(54*54 , 3)

    def forward(self, X):
        h = self.conv1(X)
        # h [batch , 1, 54, 54 ]
        h = h.squeeze(1).view(h.shape[0],-1)
        out = self.fc(h)
        return out

if __name__ == "__main__":
    model = MyConvNetwork()
    train_path = "./data/train/"
    _, dataloader = load_data(train_path, 32)
    batch = next(iter(dataloader))
    start = time.clock()
    preds = model(batch[0])
    end = time.clock()
    print(end - start)
