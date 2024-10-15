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

## 应用到 tensorboard 中
```python
from torchvision import transforms
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

dir_path = "test.jpg"
img = Image.open(dir_path)

img_to_tensor = transforms.ToTensor()
img_tensor = img_to_tensor(img)

writer = SummaryWriter("logs")
writer.add_image("test2", img_tensor)

writer.close()
```