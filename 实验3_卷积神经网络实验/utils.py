import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import torch
import torch.optim as optim
import numpy as np
def load_data(path, batch_size):
    datasets = torchvision.datasets.ImageFolder(
        root = path,
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
    )

    dataloder = DataLoader(datasets, batch_size=batch_size, shuffle=True)
    return datasets,dataloder

def get_accur(preds, labels):
    preds = preds.argmax(dim=1)
    return torch.sum(preds == labels).item()

def train(model, epochs, learning_rate, dataloader, criterion, testdataloader):
    optimizer = optim.Adam(model.parameters(),lr=learning_rate)

    train_loss_list = []
    test_loss_list = []
    train_accur_list = []
    test_accur_list = []
    train_len = len(dataloader.dataset)
    test_len = len(testdataloader.dataset)

    for i in range(epochs):
        train_loss = 0.0
        train_accur = 0
        test_loss = 0.0
        test_accur = 0
        for batch in dataloader:
            imgs, labels = batch
            preds = model(imgs)
            optimizer.zero_grad()
            loss = criterion(preds, labels)

            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_accur += get_accur(preds,labels)

        train_loss_list.append(train_loss)
        train_accur_list.append(train_accur / train_len)

        for batch in testdataloader:
            imgs, labels = batch
            preds = model(imgs)
            loss = criterion(preds, labels)
            test_loss += loss.item()
            test_accur += get_accur(preds,labels)

        test_loss_list.append(test_loss)
        test_accur_list.append(test_accur / test_len)

        print("epoch {} : train_loss : {}; train_accur : {}".format(i + 1, train_loss, train_accur / train_len))

    return np.array(train_accur_list), np.array(train_loss_list), np.array(test_accur_list), np.array(test_loss_list)