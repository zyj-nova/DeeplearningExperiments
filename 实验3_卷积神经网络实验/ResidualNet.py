import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import load_data,get_accur,train
import time

class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super().__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride,
                      padding=1, bias=False),

            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1,
                      padding=1, bias=False),
            # 尺寸不发生变化 通道改变
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        # 注意shortcut是对输入X进行卷积，利用1×1卷积改变形状
        if inchannel != outchannel or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, X):
        h = self.left(X)
        # 先相加再激活
        h += self.shortcut(X)
        out = F.relu(h)
        return out


class ResidualNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.residual_block = nn.Sequential(
            ResidualBlock(3, 32),
            ResidualBlock(32, 64),
            ResidualBlock(64, 32),
            ResidualBlock(32, 3)
        )
        self.fc1 = nn.Linear(3 * 64 * 64, 1024)
        self.fc2 = nn.Linear(1024, 3)

    def forward(self, X):
        h = self.residual_block(X)
        h = h.view(-1, 3 * 64 * 64)
        h = self.fc1(h)
        out = self.fc2(h)
        return out

if __name__ == "__main__":
    train_path = "./data/train/"
    test_path = "./data/test/"
    _, train_dataloader = load_data(train_path, 32)
    _, test_dataloader = load_data(test_path, 32)
    model = ResidualNet()
    critic = nn.CrossEntropyLoss()
    epoch = 20
    lr = 0.01
    start = time.clock()
    print("Start training model.....")
    train_accur_list, train_loss_list, test_accur_list, test_loss_list = train(model, epoch, lr, train_dataloader,
                                                                               critic, test_dataloader)
    end = time.clock()
    print("Train cost: {} s".format(end - start))
    test_accur = 0
    for batch in test_dataloader:
        imgs, labels = batch
        preds = model(imgs)
        test_accur += get_accur(preds, labels)

    print("Accuracy on test datasets : {}".format(test_accur / len(test_dataloader.dataset)))
