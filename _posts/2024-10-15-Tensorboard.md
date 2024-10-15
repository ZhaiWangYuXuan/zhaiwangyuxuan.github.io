---
title: 【pytorch笔记】Tensorboard怎么用
date: 2024-10-15 17:00:00 +0800
categories: [deep learning, pytorch]
tags: [deep learning, python, pytorch]     # TAG names should always be lowercase
description: Tensorboard怎么用
---

# 【pytorch笔记】Tensorboard怎么用

- 使用教程来自[小土堆pytorch教程](https://www.bilibili.com/video/BV1hE411t7RN)
- 配置环境：torch2.0.1+cu118与对应torchaudio和torchvision

## 引入库
```python
from torch.utils.tensorboard import SummaryWriter # 引入SummaryWriter这个类
```

## 查看转发端口
```bash
tensorboard --logdir=logs
```

## 修改端口
```bash
tensorboard --logdir=logs --port=6007
```

## 测试代码
```python
from torch.utils.tensorboard import SummaryWriter # 这是一个类
writer = SummaryWriter("logs")

# 生成y = x 的函数图像
for i in range(100):
    writer.add_scalar("y = x", i, i)

writer.close()
```

### 如何使用上述代码？
1. vscode中先运行该段代码
2. 运行后，在终端输入 `tensorboard --logdir=logs`
3. 弹出提示，跳转到浏览器查看图像

## SummaryWriter()解释
```python
mySaveDir = "Hewkick/logs"
writer = SummaryWriter(log_dir=mySaveDir)
# 最常用的形式
# 输入参数默认为log_dir=
# 即生成的文件存储在哪个位置
# 其他参数随用随查
```

## .add_scalar()方法解释
```python
writer.addscalar()
# tag (str): 图表标题
# scalar_value (float or string/blobname): 函数图像y值
# global_step (int): 函数图像x值
# 其他参数随用随查
```

- 如果想要修改函数图像，必须先**运行修改后的代码**，再输入 `tensorboard --logdir=logs`
- 如果发现图像很混乱（即同时出现了多种代码图像），那么，去自己的设定的 log_dir 文件夹，删除里面的所有文件，再运行代码，终端输入 `tensorboard --logdir=logs` 即可
