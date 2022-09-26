@[toc](pyTorch)
[官网](https://pytorch.org/get-started/locally/)
![在这里插入图片描述](https://img-blog.csdnimg.cn/50a454fecee34a10986e1279f3a97ae0.png)

# 安装
## 安装对应的CUDA和cudnn
** 以CUDA11.3为例。**
cudnn 要和cuda版本对应。

![在这里插入图片描述](https://img-blog.csdnimg.cn/c1eac10f208f4800a9ca59681cb90e3b.png)
![在这里插入图片描述](https://img-blog.csdnimg.cn/3c78702d73ff4a8bb9e3a2b030c3695d.png)


## anaconda 
### 安装
[下载地址](https://www.anaconda.com/)

### 创建并激活环境
```cpp
conda create -n pytorch
conda activate create
```
## 安装
在激活的环境中执行执行官网提供的安装指令。
```cpp
conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
```
## 测试是否安装成功

```python
import torch
import torch
print(torch.__version__)
print(torch.cuda.is_available())
1.12.1
True
```
# 官网教程- Quickstart 笔记
[教程官网](https://pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html)
## 代码
### 导入库

```python
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
```
### 下载数据

```python
# Download training data from open datasets.
training_data = datasets.FashionMNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor(),
)

# Download test data from open datasets.
test_data = datasets.FashionMNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor(),
)
```
### 数据处理
将数据传给**DataLoader**进行数据处理
```python
batch_size = 64

# Create data loaders.
train_dataloader = DataLoader(training_data, batch_size=batch_size)
test_dataloader = DataLoader(test_data, batch_size=batch_size)

for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break
```
### 创建模型
1. 创建模型类，该类继承与**nn.Module**;
2. 在**__init__**中定义需要的网络层；
3. 在**forware**中实现网络结构(前向传播）；
4. 如果GPU可用的话将模型载入GPU中

```python
# Get cpu or gpu device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")

# Define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

model = NeuralNetwork().to(device)
print(model)
```
### 损失函数和优化器

```python
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
```
### 训练
1. 模型预测；
2. 反向传播

#### 单轮训练
```python
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
```
#### 通过测试数据检测模型性能
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

#### 多轮训练

```python
epochs = 5
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")
```

## 数据
PyTorch 提供两个数据相关的类：**torch.utils.data.DataLoader**、**torch.utils.data.Dataset**  
**torch.utils.data.Dataset**: 官方提供的一些开源数据。  
**torch.utils.data.DataLoader**: 数据处理迭代器，可提供常用是数据处理操作。

## 领域库
PyTorch 提供不同领域的库：[TorchText](https://pytorch.org/text/stable/index.html)、[TorchVision](https://pytorch.org/vision/stable/index.html)、[TorchAudio](https://pytorch.org/audio/stable/index.html)

## 官方笔记学完后，需继续了解的部分
1. 查看Dataset 有哪些数据；
2. 查看dataloader提供哪些数据处理功能；
3. 查看TorchVision库提供哪些功能。

# 参考资料

> [官网](https://pytorch.org/)
> [官网教程](https://pytorch.org/tutorials/)

