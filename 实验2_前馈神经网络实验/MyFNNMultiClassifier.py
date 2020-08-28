import torch
import numpy as np
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from matplotlib import pyplot as plt

def cross_entropy_loss(preds,labels):
    return -torch.sum(torch.sum(labels * torch.log(preds),dim=1)) / len(preds)

def relu(X):
    return torch.where(X > 0, X, torch.tensor(0.0))

def softmax(z):
    # 每一行做sum
    return torch.exp(z) / torch.sum(torch.exp(z),axis=1).unsqueeze(1)
        
def get_num_correct(preds, labels):
    return (preds.argmax(dim=1) == labels).sum().item()

def L2penalty(weights):
    sum = torch.tensor(0.0)
    for weight in weights:
        sum += torch.sum(torch.square(weight))
    return  sum / 2.0

def network(X):
    H = torch.mm(X,W1) + b1
    H = relu(H)
    Y = torch.mm(H,W2) + b2
    return softmax(Y)

if __name__ == "__main__":
    train_set = torchvision.datasets.MNIST(
        root='./data'
        , train=True
        , download=False
        , transform=transforms.Compose([
            transforms.ToTensor()
        ])
    )
    test_set = torchvision.datasets.MNIST(
        root='./data',
        train=False,
        download=False,
        transform=transforms.Compose([
            transforms.ToTensor()])
    )

    train_loader = torch.utils.data.DataLoader(train_set
                                               , batch_size=512
                                               , shuffle=True
                                               )
    test_loader = torch.utils.data.DataLoader(test_set
                                              , batch_size=512
                                              , shuffle=True)

    W1 = nn.Parameter(torch.normal(0, 0.01, (28 * 28, 256)), requires_grad=True)
    b1 = nn.Parameter(torch.zeros(256), requires_grad=True)
    W2 = nn.Parameter(torch.normal(0, 0.01, (256, 10)), requires_grad=True)
    b2 = nn.Parameter(torch.zeros(10), requires_grad=True)

    params = [W1, b1, W2, b2]

    epoch = 10
    lr = 0.1
    lam = 0.1
    train_erro = []
    cnt = 1
    for i in range(epoch):
        for batch in train_loader:
            images, labels = batch
            images = images.squeeze(1).reshape(-1, 784)
            preds = network(images.type(torch.float))
            origin = nn.functional.one_hot(labels, 10)

            loss = cross_entropy_loss(preds, origin) + lam  * L2penalty([W1,W2]) / len(labels)

            loss.backward()
            
            train_erro.append(loss.item())

            for param in params:
                param.data = param - lr * param.grad
            for param in params:
                param.grad.data.zero_()
            preds = preds.argmax(dim=1)
            corr = torch.sum(preds == labels)
            print("loss :" + str(loss.item()) + "accu: " + str(corr))
            cnt += 1

    global correct
    with torch.no_grad():
        correct = 0
        for batch in test_loader:
            images, labels = batch
            images = images.squeeze(1).reshape(-1, 784)
            preds = network(images)
            correct += get_num_correct(preds, labels)
            print(correct)
    print(correct * 1.0 / len(test_set))

    train_erro = np.array(train_erro)
    
    axis = np.linspace(1,cnt + 1,cnt)
    plt.plot(axis,train_erro,color='r')
    plt.show()
