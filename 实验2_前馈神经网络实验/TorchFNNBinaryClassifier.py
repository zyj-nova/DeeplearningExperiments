import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_moons

class BinaryClassifer(nn.Module):
    def __init__(self):
        super().__init__()
        self.hid1 = nn.Linear(2,64)
        self.hid2 = nn.Linear(64,32)
        self.out = nn.Linear(32,1)
        
    def forward(self,x):
        h = self.hid1(x)
        h = F.relu(h)
        
        h = self.hid2(h)
        h = F.relu(h)
        
        o = self.out(h)
        o = torch.sigmoid(o)
        return o

if __name__ == "__main__":
    # 生成数据集
    x,y = make_moons(500,noise=0.1,shuffle=False)
    plt.scatter(x[:250,0],x[:250,1],label='0')
    plt.scatter(x[250:,0],x[250:,1],label='1')
    plt.legend()
    plt.show()

    x = torch.tensor(x).type(torch.float)
    y = torch.tensor(y).type(torch.float)
    X_train, X_test, y_train, y_test = train_test_split(x, y, shuffle=True)
    
    model = BinaryClassifer()
    lr = 0.2
    criteria = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    train_erro = []
    test_erro = []
    epoch = 100
    for i in range(epoch):
        preds = model(X_train)
        loss = criteria(preds.squeeze(1),y_train)
        print(loss.item())
        train_erro.append(loss.item())
        
        test_preds = model(X_test)
        test_erro.append(criteria(test_preds.squeeze(1),y_test).item())
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    y_hat = model(X_test)
    y_hat[y_hat >= 0.5] = 1
    y_hat[y_hat < 0.5] = 0

    print(torch.sum(y_hat.squeeze(1) == y_test).item() / len(y_test))

    train_erro = np.array(train_erro)
    test_erro = np.array(test_erro)
    axis = np.linspace(1,epoch + 1,epoch)
    plt.plot(axis,train_erro,color='r',label="train loss")
    plt.plot(axis,test_erro,label="test loss")
    plt.legend()
    plt.show()