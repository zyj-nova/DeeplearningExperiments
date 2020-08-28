import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
import torchvision
import torchvision.transforms as transforms
import numpy as np
class TorchSoftmax(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(784,10,bias = True)
    def forward(self,images):
        out = self.linear(images)
        out = torch.softmax(out,dim=1)
        return out

if __name__ == "__main__":
    train_set = torchvision.datasets.FashionMNIST(
        root='E:\\pycharm-project\\linalg-program\\FashionMNIST'
        , train=True
        , download=False
        , transform=transforms.Compose([
            transforms.ToTensor()
        ])
    )
    test_set = torchvision.datasets.FashionMNIST(
        root='E:\\pycharm-project\\linalg-program\\FashionMNIST',
        train=False,
        download=False,
        transform=transforms.Compose([
            transforms.ToTensor()])
    )
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=500, shuffle=True)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=500, shuffle=True)

    model2 = TorchSoftmax()
    optimizer = optim.SGD(model2.parameters(), lr=0.1)
    loss = nn.CrossEntropyLoss()
    epochs = 20
    losses = []
    for i in range(epochs):
        for batch in train_loader:
            images, labels = batch
            images = images.squeeze(1).reshape(-1, 784)
            preds = model2(images)
            optimizer.zero_grad()
            erro = loss(preds, labels)
            losses.append(erro.item())
            erro.backward()
            optimizer.step()
            print("loss is " + str(erro.item()))

    global correct
    with torch.no_grad():
        correct = 0
        for batch in test_loader:
            images, labels = batch
            images = images.squeeze(1).reshape(-1, 784)
            preds = model2(images)
            preds = preds.argmax(dim=1)
            correct += (preds == labels).sum()
            print(correct)
    print(correct.item() * 1.0 / len(test_set))

    x = np.arange(1, 1201)
    plt.plot(x, losses)