from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader
import numpy as np
import torch
from matplotlib import pyplot as plt

def train_test_split(window_size):
    data = np.load("./PEMS04.npz")
    pem04 = data['data']
    scaler_data = np.zeros((pem04.shape[0], pem04.shape[1], pem04.shape[2]))

    v_scaler = MinMaxScaler()  # 速度归一化
    o_scaler = MinMaxScaler()  # 拥挤程度归一化
    f_scaler = MinMaxScaler()  # 车流量归一化

    scaler_data[:, :, 0] = f_scaler.fit_transform(pem04[:, :, 0])  # 车流量
    scaler_data[:, :, 1] = o_scaler.fit_transform(pem04[:, :, 1])  # 拥挤程度
    scaler_data[:, :, 2] = v_scaler.fit_transform(pem04[:, :, 2])  # 速度

    ratio = int(pem04.shape[0] * 0.75)
    train_data = scaler_data[:ratio, :, :]
    test_data = scaler_data[ratio:, :, :]

    # 训练集数据时间序列采样
    result = []
    for i in range(len(train_data) - window_size - 1):
        tmp = train_data[i: i + window_size, :, :]
        tmp = tmp.reshape(-1, 307 * 3)
        # 后1min的数据作为label
        label = train_data[i + window_size + 1, :, :].reshape(1, -1)
        tmp = np.concatenate((tmp, label), axis=0)
        result.append(tmp)

    train_loader = DataLoader(result, batch_size=30, shuffle=False)

    test_sets = []

    for i in range(len(test_data) - window_size - 1):
        tmp = test_data[i: i + window_size, :, :]
        tmp = tmp.reshape(-1, 307 * 3)
        # 后1min的数据作为label
        label = test_data[i + window_size + 1, :, :].reshape(1, -1)
        tmp = np.concatenate((tmp, label), axis=0)
        test_sets.append(tmp)
    test_loader = DataLoader(test_sets, batch_size=36, shuffle=False)
    # 返回MinMaxScaler以便反归一化
    return train_loader, test_loader, v_scaler, f_scaler, o_scaler

def train(epoch, train_loader, test_loader, model, optimizer, window_size, criterion,alt):
    train_loss_list = []
    test_loss_list = []
    for i in range(epoch):
        losses = 0
        for batch in train_loader:
            # [src_len, batch, embedded]
            batch = batch.permute(1, 0, 2)
            x = batch[:window_size, :, :].type(torch.float)
            label = batch[-1, :, :].type(torch.float)
            if alt == "my":
                pred, _ = model(x)
                pred = pred[-1, :, :]
            else:
                pred = model(x)
                pred = pred.squeeze(0)
            loss = criterion(pred, label)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses += loss.item()
        train_loss_list.append(losses)
        
        print("epoch: {},loss : {}".format(i + 1, losses))
        #测试集上的loss
        losses = 0
        for batch in test_loader:
            batch = batch.permute(1, 0, 2)
            x = batch[:window_size, :, :].type(torch.float)
            label = batch[-1, :, :].type(torch.float)
            if alt == "my":
                pred, _ = model(x)
                pred = pred[-1, :, :]
            else:
                pred = model(x)
                pred = pred.squeeze(0)
            loss = criterion(pred, label)
            losses += loss.item()
        test_loss_list.append(losses)
        
    return train_loss_list, test_loss_list

def test(model, test_loader,window_size,criterion, alt):
    model.eval()
    test_loss = 0.0
    test_preds = []
    test_label = []
    for batch in test_loader:
        batch = batch.permute(1, 0, 2)
        x = batch[:window_size, :, :].type(torch.float)
        label = batch[-1, :, :].type(torch.float)
        if alt == "my":
            pred, _ = model(x)
            pred = pred[-1, :, :]
        else:
            pred = model(x)
            pred = pred.squeeze(0)
        test_preds.append(pred)
        test_label.append(label)
        loss = criterion(pred, label)
        test_loss += loss.item()
    print("test loss: {}".format(test_loss))
    return test_preds, test_label

def show_preds(batch, test_preds, test_label, scaler, type, no):
    # 绘制第 no 个探头的速度、车流量、拥挤程度的对比
    # 获取前batch的预测数据
    p = torch.stack(test_preds[:batch], dim=0).reshape(-1, 307, 3)
    t = torch.stack(test_label[:batch], dim=0).reshape(-1, 307, 3)

    predict = scaler.inverse_transform(p[:, :, type].detach().numpy())
    labels = scaler.inverse_transform(t[:, :, type].detach().numpy())
    no_predict = predict[:, no]
    no_labels = labels[:, no]

    x = np.arange(1, p.shape[0] + 1)

    plt.plot(x, no_predict, label='predict')
    plt.plot(x, no_labels, label="original")
    plt.legend()
    plt.show()
