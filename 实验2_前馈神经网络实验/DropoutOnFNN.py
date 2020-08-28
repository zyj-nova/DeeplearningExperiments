import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from torchvision import transforms

def dropout(x, keep_prob = 0.5):
    '''
    np.random.binomial 当输入二维数组时，按行按列（每个维度）都是按照给定概率生成1的个数，
比如 输入 10 * 6的矩阵，按照0.5的概率生成1 那么每列都大概会有5个1，每行大概会有3个1，
其实就不用考虑按行drop或者按列drop，相当于每行生成的mask都是不一样的，那么矩阵中每行的元素（代表一层中的神经元）都是按照不同的mask失活的
当矩阵形状改变行列代表的意义不一样时，由于每行每列（各个维度）的1的个数都是按照prob留存的，因此对结果没有影响。
    '''
    mask = torch.from_numpy(np.random.binomial(1,keep_prob,x.shape))
    return x * mask / keep_prob


class Network2(nn.Module):
    def __init__(self ,input_dim ,out_dim,keep_prob = 0.5):
        super().__init__()
        self.layer1 = nn.Sequential(  # 全连接层     [1, 28, 28]
            nn.Linear(input_dim, 400),       # 输入维度，输出维度
            nn.BatchNorm1d(400),  # 批标准化，加快收敛，可不需要
            nn.ReLU() , 				 # 激活函数
            nn.Dropout(1 - keep_prob)
        )

        self.layer2 = nn.Sequential(
            nn.Linear(400, 200),
            nn.BatchNorm1d(200),
            nn.ReLU(),
            nn.Dropout(1 - keep_prob)
        )

        self.layer3 = nn.Sequential(   # 全连接层
            nn.Linear(200, 100),
            nn.BatchNorm1d(100),
            nn.ReLU(),
            nn.Dropout(1 - keep_prob)
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
def get_num_correct(preds, labels):
    return (preds.argmax(dim=1) == labels).sum().item()

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

    net = Network2(28 * 28, 10,keep_prob= 0.7)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
    criterion = nn.CrossEntropyLoss()
    epoch = 10
    train_erro = []
    test_erro = []
    cnt = 1
    for i in range(epoch):
        train_accur = 0.0
        train_loss = 0.0
        
        for batch in train_loader:
            images, labels = batch
            #images, labels = images.to(device), labels.to(device)
            images = images.squeeze(1).reshape(images.shape[0], -1)
            preds = net(images)
            optimizer.zero_grad()
            loss = criterion(preds, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_accur += get_num_correct(preds, labels)
            
            train_erro.append(loss.item())
            cnt += 1
            #在测试集上的loss
    #         net.eval()
    #         test_preds = net(test_set.data.type(torch.float).reshape(-1,784))
    #         test_erro.append(criterion(test_preds,test_set.targets).item())
            
        print("loss :" + str(train_loss) + ",train accur:" + str(train_accur * 1.0 / 60000))
    
    global correct
    with torch.no_grad():
        correct = 0
        for batch in test_loader:
            images, labels = batch
            #images, labels = images.to(device), labels.to(device)
            images = images.squeeze(1).reshape(-1, 784)
            net.eval()
            preds = net(images)

            preds = preds.argmax(dim=1)
            correct += (preds == labels).sum()
            #print(correct)
    print(correct.item() * 1.0 / len(test_set))

