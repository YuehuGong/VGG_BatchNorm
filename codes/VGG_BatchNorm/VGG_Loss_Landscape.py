import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from torch import nn
import numpy as np
import torch
import os
import random
from tqdm import tqdm
from IPython import display

from models.vgg import VGG_A, VGG_A_BatchNorm
from data.loaders import get_cifar_loader

# ----------------- 配置路径 -----------------
module_path = os.path.dirname(os.getcwd())
home_path = module_path

figures_path = os.path.join(home_path, 'reports', 'figures')
loss_save_path = os.path.join(home_path, 'reports', 'losses')
grad_save_path = os.path.join(home_path, 'reports', 'grads')

# 确保目录存在
for path in [figures_path, loss_save_path, grad_save_path]:
    os.makedirs(path, exist_ok=True)

# ----------------- 设备配置 -----------------
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))

# ----------------- 数据加载 -----------------
train_loader = get_cifar_loader(train=True)
val_loader = get_cifar_loader(train=False)

# ----------------- 计算准确率 -----------------
def get_accuracy(model, data_loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in data_loader:
            x, y = x.to(device), y.to(device)
            outputs = model(x)
            _, predicted = torch.max(outputs.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
    return correct / total

# ----------------- 设置随机种子 -----------------
def set_random_seeds(seed_value=0, device='cpu'):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    random.seed(seed_value)
    if device != 'cpu':
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# ----------------- 训练函数 -----------------
def train(model, optimizer, criterion, train_loader, val_loader, epochs_n=100, model_name='model'):
    model.to(device)

    learning_curve = []
    train_accuracy_curve = []
    val_accuracy_curve = []

    batches_n = len(train_loader)
    losses_list = []
    grads = []

    for epoch in tqdm(range(epochs_n), unit='epoch', desc=f'Training {model_name}'):
        model.train()
        epoch_loss = 0
        loss_list = []
        grad_list = []

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            prediction = model(x)
            loss = criterion(prediction, y)

            loss.backward()
            optimizer.step()

            loss_list.append(loss.item())
            epoch_loss += loss.item()

            # 如果需要梯度，可以在这里保存 grad，比如
            # grad_list.append(model.classifier[4].weight.grad.clone().cpu().numpy())

        epoch_loss /= batches_n
        learning_curve.append(epoch_loss)
        losses_list.append(loss_list)
        grads.append(grad_list)

        train_acc = get_accuracy(model, train_loader)
        val_acc = get_accuracy(model, val_loader)
        train_accuracy_curve.append(train_acc)
        val_accuracy_curve.append(val_acc)

        # 绘制训练曲线
        display.clear_output(wait=True)
        fig, axes = plt.subplots(1, 2, figsize=(15, 4))
        axes[0].plot(learning_curve, label='Training Loss')
        axes[0].set_title(f'{model_name} Training Loss')
        axes[0].legend()
        axes[1].plot(train_accuracy_curve, label='Train Accuracy')
        axes[1].plot(val_accuracy_curve, label='Validation Accuracy')
        axes[1].set_title(f'{model_name} Accuracy')
        axes[1].legend()
        plt.savefig(os.path.join(figures_path, f'{model_name}_curve_epoch_{epoch+1}.png'))
        plt.close()

    return losses_list, grads, learning_curve, train_accuracy_curve, val_accuracy_curve

# ----------------- 主程序 -----------------
if __name__ == '__main__':
    set_random_seeds(seed_value=2020, device=device.type)

    models = {
        'VGG_A': VGG_A(),
        'VGG_A_BatchNorm': VGG_A_BatchNorm()
    }

    epo = 20
    lr = 0.001
    criterion = nn.CrossEntropyLoss()

    all_losses = {}
    all_grads = {}
    all_min_curves = {}
    all_max_curves = {}

    for name, model in models.items():
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        losses, grads, loss_curve, train_acc_curve, val_acc_curve = train(
            model, optimizer, criterion, train_loader, val_loader, epochs_n=epo, model_name=name
        )

        # 保存loss和grad数据
        np.savetxt(os.path.join(loss_save_path, f'{name}_loss.txt'), np.array(losses), fmt='%s')
        np.savetxt(os.path.join(grad_save_path, f'{name}_grads.txt'), np.array(grads), fmt='%s')

        # 计算step级别最大最小loss，用于loss landscape绘图
        min_curve = []
        max_curve = []
        for step_losses in zip(*losses):
            min_curve.append(min(step_losses))
            max_curve.append(max(step_losses))

        all_losses[name] = loss_curve
        all_min_curves[name] = min_curve
        all_max_curves[name] = max_curve
        all_grads[name] = grads

    # 画两模型的loss landscape放在一张图
    plt.figure(figsize=(10, 6))
    x = list(range(len(all_min_curves['VGG_A'])))
    for name in models.keys():
        plt.fill_between(x, all_min_curves[name], all_max_curves[name], alpha=0.3, label=f'{name} Loss Range')
        plt.plot(all_min_curves[name], label=f'{name} Min Loss')
        plt.plot(all_max_curves[name], label=f'{name} Max Loss')

    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Loss Landscape Comparison")
    plt.legend()
    plt.savefig(os.path.join(figures_path, 'loss_landscape_comparison.png'))
    plt.close()
