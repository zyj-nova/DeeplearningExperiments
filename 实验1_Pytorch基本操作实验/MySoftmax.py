import torch
from matplotlib import pyplot as plt
import torchvision
import torchvision.transforms as transforms
from torch.nn import functional as F

def softmax(z):
    # 每一行做sum
    return torch.exp(z) / torch.sum(torch.exp(z),axis=1).unsqueeze(1)

def cross_entropy_loss(preds,labels):
    return -torch.sum(labels * torch.log(preds)) / len(preds)

def model(X,w):
    return softmax(torch.mm(X,w))

if __name__ == "__main__":

    train_set = torchvision.datasets.FashionMNIST(
        root='E:\\pycharm-project\\linalg-program\\FashionMNIST'
        ,train=True
        ,download=False
        ,transform=transforms.Compose([
            transforms.ToTensor()
        ])
    )
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=1000, shuffle=True)

    epochs = 20
    lr = 0.1
    # 初始化权重 输入图片维度 [batch_size, 1, 28, 28]
    w = torch.normal(0, 1, (784, 10))
    w.requires_grad_(requires_grad=True)
    ans = []
    for i in range(epochs):
        for batch in train_loader:
            images, labels = batch
            images = images.squeeze(1).reshape(-1, 784)
            preds = model(images, w)
            labels = F.one_hot(labels, 10)
            # e = loss(preds,labels)
            e = cross_entropy_loss(preds, labels)
            ans.append(e)
            e.backward()
            w.data -= lr * w.grad
            # 清空梯度
            w.grad.data.zero_()
            print("batch erro:" + str(e.item()))

    test_set = torchvision.datasets.FashionMNIST(
        root='./data',
        train=False,
        download=False,
        transform=transforms.Compose([
            transforms.ToTensor()])
    )

    test_loader = torch.utils.data.DataLoader(test_set, batch_size=500, shuffle=True)
    global correct
    with torch.no_grad():
        correct = 0
        for batch in test_loader:
            images, labels = batch
            images = images.squeeze(1).reshape(-1, 784)
            preds = model(images, w)
            preds = preds.argmax(dim=1)
            correct += (preds == labels).sum()
            print(correct)
    print(correct.item() * 1.0 / len(test_set))

    ans2 = []
    for a in ans:
        ans2.append(a.item())

    x = np.arange(1, 1201)
    plt.plot(x, ans2)
