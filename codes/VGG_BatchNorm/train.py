import torch
import torch.nn as nn
import torch.optim as optim
from data.loaders import get_cifar_loader  # 你的本地加载器
from models.vgg import VGG_A, VGG_A_Light, VGG_A_Dropout, VGG_A_BatchNorm, VGG_A_Dropout_BN, VGG_A_Dropout_BN_Res, VGG_A_LeakyReLU, VGG_A_Dropout_BN_Res_Leaky

import os
from tqdm import tqdm

# 配置参数
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
batch_size = 1024
num_epochs = 30
learning_rate = 0.001
model_name = 'VGG_A'  # 可以改成 'VGG_A_Light' 或 'VGG_A_Dropout'

use_l2 = False  # 是否使用L2正则化（weight_decay）
optimizer_name = 'Adam'  # 'Adam' 或 'SGD' 或其他你想支持的

# 选择模型（方便改）
def get_model(name='VGG_A'):
    if name == 'VGG_A':
        return VGG_A()
    elif name == 'VGG_A_Light':
        return VGG_A_Light()
    elif name == 'VGG_A_Dropout':
        return VGG_A_Dropout()
    elif name == 'VGG_A_BatchNorm':
        return VGG_A_BatchNorm()
    elif name == 'VGG_A_Dropout_BN':
        return VGG_A_Dropout_BN()
    elif name == 'VGG_A_Dropout_BN_Res':
        return VGG_A_Dropout_BN()
    elif name == 'VGG_A_LeakyReLU':
        return VGG_A_Dropout_BN()
    elif name == 'VGG_A_Dropout_BN_Res_Leaky':
        return VGG_A_Dropout_BN()
    else:
        raise ValueError(f"Unknown model name {name}")


# 选择优化器
def get_optimizer(name, parameters, lr, weight_decay):
    if name.lower() == 'adam':
        return optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
    elif name.lower() == 'sgd':
        return optim.SGD(parameters, lr=lr, momentum=0.9, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer {name}")

# 训练函数
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    avg_loss = running_loss / total
    accuracy = 100. * correct / total
    return avg_loss, accuracy

# 验证函数
def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    avg_loss = running_loss / total
    accuracy = 100. * correct / total
    return avg_loss, accuracy

def main():
    # 选模型名，方便切换
    model = get_model(model_name).to(device)

    train_loader = get_cifar_loader(root='../data/', batch_size=batch_size, train=True, shuffle=True)
    val_loader = get_cifar_loader(root='../data/', batch_size=batch_size, train=False, shuffle=False)

    criterion = nn.CrossEntropyLoss()

    # L2正则化权重衰减设置
    weight_decay = 1e-4 if use_l2 else 0.0

    optimizer = get_optimizer(optimizer_name, model.parameters(), learning_rate, weight_decay)

    best_acc = 0.0
    save_path = f'best_{model_name}.pth'

    for epoch in tqdm(range(1, num_epochs + 1)):
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        print(f"Epoch {epoch}/{num_epochs}:")
        print(f"  Train loss: {train_loss:.4f}, Train acc: {train_acc:.2f}%")
        print(f"  Val   loss: {val_loss:.4f}, Val   acc: {val_acc:.2f}%")

        # 保存最好模型
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"  Saved best model with val acc: {best_acc:.2f}%")

if __name__ == '__main__':
    main()
