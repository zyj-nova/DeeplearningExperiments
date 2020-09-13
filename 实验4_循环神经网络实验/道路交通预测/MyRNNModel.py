import torch
import torch.nn as nn
import torch.optim as optim
from utils import train_test_split, train, test,show_preds

class MyRNNModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.i_h = nn.Parameter(torch.normal(0, 0.01, (input_size, hidden_size)), requires_grad=True)
        self.h_h = nn.Parameter(torch.normal(0, 0.01, (hidden_size, hidden_size)), requires_grad=True)
        self.h_o = nn.Parameter(torch.normal(0, 0.01, (hidden_size, output_size)), requires_grad=True)

        self.b_i = nn.Parameter(torch.zeros(hidden_size), requires_grad=True)
        self.b_o = nn.Parameter(torch.zeros(output_size), requires_grad=True)

        self.tanh = nn.Tanh()
        self.leaky_relu = nn.LeakyReLU()

    def forward(self, x):
        '''
            x: [src_len, batch_size, embedding]
        '''
        batch_size = x.shape[1]
        seq_len = x.shape[0]
        # 初始隐藏态 每个时间步的隐藏态：[1, batch, hidden ]
        h = torch.normal(0, 0.01, (batch_size, self.hidden_size))
        outputs = []
        # 时间步内循环
        for i in range(seq_len):
            # 第i个时间步 x[i,:,:]是个二维张量
            # 矩阵和向量相加，每行加偏置
            h = torch.matmul(x[i, :, :], self.i_h) + torch.matmul(h, self.h_h) + self.b_i
            h = self.tanh(h)
            # h: [batch, hidden]
            o = self.leaky_relu(torch.matmul(h, self.h_o) + self.b_o)
            outputs.append(o)
        # out [seq_len, batch, output_size]
        return torch.stack(outputs, dim=0), h

if __name__ == "__main__":
    # 训练
    criterion = nn.MSELoss()
    window_size = 6
    epoch = 20
    lr = 0.1
    model = MyRNNModel(3 * 307, 256, 3 * 307)
    optimizer = optim.SGD(model.parameters(), lr=lr)
    train_loader, test_loader , v_scaler, f_scaler, o_scaler = train_test_split(window_size)

    train_loss_list, test_loss_list = train(epoch,train_loader,test_loader,model,optimizer,window_size,criterion,"my")

    test_preds, test_label = test(model,test_loader,window_size,criterion,"my")

    show_preds(15,test_preds,test_label,v_scaler,2,1)

    #绘制loss曲线变化图
    epoches = np.arange(1, epoch + 1)
    train_loss_list = np.array(train_loss_list)
    test_loss_list = np.array(test_loss_list)

    plt.plot(epoches, train_loss_list, label = "train")
    plt.plot(epoches, test_loss_list, label = "test")
    plt.legend()
    plt.plot()