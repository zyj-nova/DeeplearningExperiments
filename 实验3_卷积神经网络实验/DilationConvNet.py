import torch.nn as nn
import numpy as np

from matplotlib import pyplot as plt
import time
from utils import get_accur,load_data,train

class ConvNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=2, padding=0, dilation=1),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=0, dilation=2),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5, stride=1, padding=0, dilation=5),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.fc = nn.Linear(128 * 3 * 3, 3)

    def forward(self, x):
        x = self.layer1(x)

        x = self.layer2(x)

        x = self.layer3(x)

        x = x.view(-1, 128 * 3 * 3)

        out = self.fc(x)

        return out


if __name__ == "__main__":
    train_path = "/home/zyj/Downloads/车辆分类数据集/train/"
    test_path = "/home/zyj/Downloads/车辆分类数据集/test/"
    train_datasets, train_dataloader = load_data(train_path, 64)
    test_datasets, test_dataloader = load_data(test_path, 64)
    model = ConvNetwork()
    critic = nn.CrossEntropyLoss()
    epoch = 15
    lr = 0.01
    start = time.clock()
    train_accur_list, train_loss_list, test_accur_list, test_loss_list = train(model, epoch, lr, train_dataloader,
                                                                               critic, test_dataloader)
    end = time.clock()
    test_accur = 0
    for batch in test_dataloader:
        imgs, labels = batch
        preds = model(imgs)
        test_accur += get_accur(preds, labels)

    print("Accuracy on test datasets : {}".format(test_accur / len(test_datasets)))
    print("Total time".format(end - start))
    x_axis = np.arange(1, epoch + 1)
    plt.plot(x_axis, train_loss_list, label="train loss")
    plt.plot(x_axis, test_loss_list, label="test loss")
    plt.legend()
    plt.show()

    plt.plot(x_axis, train_accur_list, label="train accur")
    plt.plot(x_axis, test_accur_list, label="test accur")
    plt.legend()
    plt.show()
