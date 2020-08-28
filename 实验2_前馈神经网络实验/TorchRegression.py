import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class RegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.hid1 = nn.Linear(1, 32)
        self.hid2 = nn.Linear(32, 1)

    def forward(self, x):
        h = self.hid1(x)
        h = F.relu(h)
        out = self.hid2(h)
        return out

if __name__ == "__main__":
    criti = nn.MSELoss()
    lr = 0.1
    net = RegressionModel()
    optimizer = optim.SGD(net.parameters(), lr=lr)

    for i in range(200):
        preds = net(x)
        loss = criti(preds, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(loss.item())

    preds = net(x)
    preds = preds.detach().numpy()
    plt.plot(x, preds, color='r')
    plt.scatter(x, y)
    plt.show()