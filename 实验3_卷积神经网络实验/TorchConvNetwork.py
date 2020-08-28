import torch.nn as nn
from utils import get_accur,load_data,train
from matplotlib import pyplot as plt
import numpy as np


class ConvNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=7, stride=2, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=2, padding=0),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=3, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(3),
            nn.ReLU()
        )
        self.out = nn.Linear(in_features=3 * 9 * 9, out_features=3, bias=True)

    def forward(self, x):
        x = self.layer1(x)

        x = self.layer2(x)

        x = self.layer3(x)
        x = self.layer4(x)

        x = x.reshape(x.shape[0], -1)
        out = self.out(x)

        return out


if __name__ == "__main__":
    train_path = "./data/train/"
    test_path = "./data/test/"
    train_datasets, train_dataloader = load_data(train_path, 64)
    test_datasets, test_dataloader = load_data(test_path, 64)
    model = ConvNetwork()
    critic = nn.CrossEntropyLoss()
    epoch = 20
    lr = 0.01
    train_accur_list, train_loss_list, test_accur_list, test_loss_list = train(model,epoch, lr ,train_dataloader, critic, test_dataloader)

    test_accur = 0
    for batch in test_dataloader:
        imgs, labels = batch
        preds = model(imgs)
        test_accur += get_accur(preds,labels)

    print("Accuracy on test datasets : {}".format(test_accur / len(test_datasets)))

    x_axis = np.arange(1, epoch + 1)
    plt.plot(x_axis, train_loss_list, label="train loss")
    plt.plot(x_axis, test_loss_list, label = "test loss")
    plt.legend()
    plt.show()

    plt.plot(x_axis, train_accur_list, label = "train accur")
    plt.plot(x_axis, test_accur_list, label = "test accur")
    plt.legend()
    plt.show()
    

