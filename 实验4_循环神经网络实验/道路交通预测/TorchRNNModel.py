import torch
import torch.nn as nn
import torch.optim as optim
from utils import train_test_split, train, test,show_preds
import torch.optim as optim
from matplotlib import pyplot as plt

class RNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        # 注意rnn cell并没有h_w这个矩阵
        super().__init__()
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size)
        self.output = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        '''
            x: [src_len, batch_size, embedding]
        '''
        _, hidden = self.rnn(x)

        out = self.output(hidden)

        return out


if __name__ == "__main__":
    criterion = nn.MSELoss()
    window_size = 6
    epoch = 20
    lr = 0.1
    model = RNNModel(3 * 307, 256, 3 * 307)
    optimizer = optim.SGD(model.parameters(), lr=lr)
    train_loader, test_loader , v_scaler, f_scaler, o_scaler = train_test_split(window_size)

    train_loss_list, test_loss_list = train(epoch,train_loader,test_loader, model,optimizer,window_size,criterion, "torch")
    test_preds, test_label = test(model,test_loader,window_size,criterion,"torch")
    
    show_preds(15,test_preds,test_label,o_scaler,1,0)

    #绘制loss曲线变化图
    epoches = np.arange(1, epoch + 1)
    train_loss_list = np.array(train_loss_list)
    test_loss_list = np.array(test_loss_list)

    plt.plot(epoches, train_loss_list, label = "train")
    plt.plot(epoches, test_loss_list, label = "test")
    plt.legend()
    plt.plot()
