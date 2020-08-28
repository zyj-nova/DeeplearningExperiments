import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
    a = torch.tensor([[1,2,3]])
    b = torch.tensor([[2],[3]])
    print(a - b)
    print(a.sub(b))
    print(torch.sub(a,b))

    #################################

    p = torch.normal(0,0.01,(3,2))
    q = torch.normal(0, 0.01, (4, 2))
    q_t = q.t()
    res = torch.matmul(p,q_t)
    print(res)

    ###################################
    x = torch.tensor([1.0], requires_grad=True)
    y1 = x ** 2


    with torch.no_grad():
        y2 = x ** 3
    y3 = y1 + y2
    y3.backward()
    print(x.grad)
    # 输出为2，而正确答案应该是5。原因是y2被包裹在no-grad代码块中，并反向传播中并没有计算对x的梯度，

    ######################################




