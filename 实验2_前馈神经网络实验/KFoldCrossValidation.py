import torch
import torchvision
from torchvision import transforms
import torch.nn as nn
from torch.utils.data import DataLoader,Dataset   
import torch.nn.functional as F

def get_num_correct(preds, labels):
    return (preds.argmax(dim=1) == labels).sum().item()

class TrainDatasets(Dataset):
    def __init__(self,train_features,train_labels):
        
        self.x_data = train_features
        self.y_data = train_labels
        self.len = len(train_labels)
    
    def __getitem__(self,index):
        return self.x_data[index],self.y_data[index]
    
    def __len__(self):
        return self.len
        
class Network(nn.Module):
    def __init__(self ,input_dim,out_dim):
        super().__init__()
        self.layer1 = nn.Sequential(  # 全连接层     [1, 28, 28]
            nn.Linear(input_dim, 400),       # 输入维度，输出维度
            nn.BatchNorm1d(400),  # 批标准化，加快收敛，可不需要
            nn.ReLU()  				 # 激活函数
        )

        self.layer2 = nn.Sequential(
            nn.Linear(400, 200),
            nn.BatchNorm1d(200),
            nn.ReLU()
        )

        self.layer3 = nn.Sequential(   # 全连接层
            nn.Linear(200, 100),
            nn.BatchNorm1d(100),
            nn.ReLU()
        )

        self.layer4 = nn.Sequential(   # 最后一层为实际输出，不需要激活函数，因为有 10 个数字，所以输出维度为 10，表示10 类
            nn.Linear(100, out_dim),
        )

    def forward(self ,x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        output = self.layer4(x)
        return output
    
########k折划分############        
def get_k_fold_data(k, i, X, y):  ###此过程主要是步骤（1）
    # 返回第i折交叉验证时所需要的训练和验证数据，分开放，X_train为训练数据，X_valid为验证数据
    assert k > 1
    fold_size = X.shape[0] // k  # 每份的个数:数据总条数/折数（组数）
    
    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)  #slice(start,end,step)切片函数
        ##idx 为每组 valid
        X_part, y_part = X[idx, :], y[idx]
        if j == i: ###第i折作valid
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = torch.cat((X_train, X_part), dim=0) #dim=0增加行数，竖着连接
            y_train = torch.cat((y_train, y_part), dim=0)
    #print(X_train.size(),X_valid.size())
    return X_train, y_train, X_valid,y_valid

def k_fold(k, X, y, num_epoches = 4, learning_rate = 0.1, weight_decay = 0.01, batch_size = 512):
    train_loss_sum, valid_loss_sum = 0, 0
    train_acc_sum ,valid_acc_sum = 0,0
    model = Network(28*28, 10)
    optimizer = torch.optim.SGD(params=model.parameters(), lr= learning_rate, weight_decay=weight_decay)
    loss_func = nn.CrossEntropyLoss()
    # k 折交叉验证
    for i in range(k):
        X_train, y_train, X_valid,y_valid = get_k_fold_data(k, i, X, y)
        datasets = TrainDatasets(X_train, y_train)
        train_dataloader = DataLoader(datasets, batch_size=batch_size, shuffle=True)
        fold_loss = 0.0
        fold_accur = 0.0
        for epoch in range(num_epoches):
            train_accur = 0.0
            train_loss = 0.0
            for batch in train_dataloader:  ###分批训练
                images,labels = batch
                images = images.type(torch.float).squeeze(1).reshape(images.shape[0], -1)
                preds = model(images)
                optimizer.zero_grad()
                loss = loss_func(preds,labels)
                loss.backward()
                optimizer.step()
                
                # 记录train_loss
                train_loss += loss.item()
                train_accur += get_num_correct(preds, labels)
            fold_loss += train_loss
            fold_accur += train_accur * 1.0 / 60000
            print("train loss :" + str(train_loss) + ", train accur: " + str(train_accur * 1.0 / 60000))
            # 在验证集上的loss和准确率
            valid_preds = model(X_valid.type(torch.float).squeeze(1).reshape(X_valid.shape[0],-1))
            valid_loss = loss_func(valid_preds,y_valid)
            valid_accur = get_num_correct(valid_preds,y_valid)
            print("valid loss : " + str(valid_loss.item()) + ", valid accur: " + str(valid_accur * 1.0 /  len(y_valid)))
        print("# " + str(i) + "折结果，average_loss : " + str(fold_loss / num_epoches) + " average_accur : " + str(fold_accur / num_epoches))
    global correct
    with torch.no_grad():
        correct = 0
        for batch in test_loader:
            images, labels = batch
            #images, labels = images.to(device), labels.to(device)
            images = images.type(torch.float).squeeze(1).reshape(-1, 784)
            preds = model(images)

            preds = preds.argmax(dim=1)
            correct += (preds == labels).sum()
            print(correct)
        print(correct.item() * 1.0 / len(test_set))


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
    k_fold(10,train_set.data,train_set.targets)