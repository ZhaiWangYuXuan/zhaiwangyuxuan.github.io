---
title: 【pytorch笔记】Transformers怎么用
date: 2024-10-15 20:00:00 +0800
categories: [deep learning, pytorch]
tags: [deep learning, python, pytorch]     # TAG names should always be lowercase
description: Transformers怎么用
---

## 前言
- 使用教程来自[小土堆pytorch教程](https://www.bilibili.com/video/BV1hE411t7RN)
- 配置环境：torch2.0.1+cu118与对应torchaudio和torchvision

## 介绍

### transformers库是做什么的
- 将 transformers 这个库看作是一个工具箱，我们借由此工具箱生成对象，我们输入图片，并让该对象以特定格式输出

### 引入库
```python
from torchvision import transforms
from PIL import Image
```

## 如何使用 transformers 库
```python
dir_path = "test.jpg"
img = Image.open(dir_path)

img_to_tensor = transforms.ToTensor() # 构建对象
img_tensor = img_to_tensor(img) # 输入对象即可
```

## tensor 是什么
- Tensor（张量） 是一种数据结构，被广泛用于神经网络模型中，用来表示或者编码神经网络模型的输入、输出和模型参数等等

## 常见的 transformers 用法

### 应用到 tensorboard 中
```python
from torchvision import transforms
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

dir_path = "test.jpg"
img = Image.open(dir_path)

img_to_tensor = transforms.ToTensor()
img_tensor = img_to_tensor(img) # 转换成 CHW 

writer = SummaryWriter("logs")
writer.add_image("test2", img_tensor)

writer.close()
```

### __call__ 办法
假设给定一个类
```python
class person:
    
    def __call__(self, name):
        print("Hello," + " " + name + "!!!!!!!")
```

在另一个py文件中调用

**这样子就直接()调用即可**
```python
from myClass import person
omor = person()
omor("hhh")
```

### transformers.Compose() 类
```python
# 为什么是224*224 懂得都懂 不懂的我也不好多说
myTransform = transforms.Compose([
    transforms.Resize((224, 224)),  # 调整到224*224
    transforms.ToTensor()  # 转换为张量
])
```

### ToTensor() 类
```python
tran_totensor = transforms.ToTensor()
pic_tensor = tran_totensor(img)
# Convert a PIL Image or ndarray to tensor and scale the values accordingly
# 将 PIL 图像或 ndarray 转换为张量，并相应地缩放值。
```

### Normalize 类
```python
tran_normal = transforms.Normalize([1, 3, 5], [1, 3, 5])
res = tran_normal(pic_tensor)
```

Given mean: ``(mean[1],...,mean[n])`` and std: ``(std[1],..,std[n])`` for ``n``channels, this transform will normalize each channel of the input

output\[channel\] = (input\[channel\] - mean\[channel\]) / std\[channel\]

**归一化，使得范围归一到\[-1, 1\]**