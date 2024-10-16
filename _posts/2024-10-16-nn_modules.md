---
title: 【pytorch笔记】nn.modules怎么用
date: 2024-10-16 15:00:00 +0800
categories: [deep learning, pytorch]
tags: [deep learning, python, pytorch]     # TAG names should always be lowercase
description: nn.modules怎么用
---

## 前言
- 使用教程来自[小土堆pytorch教程](https://www.bilibili.com/video/BV1hE411t7RN)
- 配置环境：torch2.0.1+cu118与对应torchaudio和torchvision

## 引入库
```python
import torch.nn as nn
```

## 实例代码01 - 官方代码
```python
# Base class for all neural network modules.
# Your models should also subclass this class.
class Model(nn.Module): # 必须继承 nn.module 这个父类
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 20, 5)
        self.conv2 = nn.Conv2d(20, 20, 5)

    # forward 函数
    # 输入 input 用来输出 output 
    def forward(self, x):
        x = F.relu(self.conv1(x))
        return F.relu(self.conv2(x))
    
    # input - conv1 - relu - conv2 - relu - output
    # 输入  - 卷积   - 非线性 - 卷积 - 非线性 - 输出
```

> 必须继承 `nn.module` 这个父类
{: .prompt-info }

## 实例代码02 - 自用测试
```python
class test(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        output = input + 1
        return output
    
mymodel = test()
test_x = torch.tensor(1.0)
output_x = mymodel(test_x)
print(output_x)
```

- 其实没啥说的，实现一个简单的 +1 操作，不过有了 `forward()` 办法后，直接用括号就可以，不用 `.forward()` 了