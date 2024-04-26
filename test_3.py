from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import random
from torchvision.models.resnet import ResNet101_Weights
import torch.optim as optim
import torch.nn as nn
from torchvision import models
import os
import torch
import numpy as np
import matplotlib.pyplot as plt


def prepare_datasets(source_dir):
    # 数据变换设置
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 调整图像大小至224x224
        transforms.ToTensor(),  # 转换为Tensor
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 归一化
    ])

    # 创建数据集
    dataset = datasets.ImageFolder(root=source_dir, transform=transform)

    # 创建索引的映射以便按类分割数据
    targets = torch.tensor(dataset.targets)
    classes = sorted(set(targets.numpy()))

    train_indices = []
    val_indices = []
    test_indices = []

    for c in classes:
        indices = (targets == c).nonzero(as_tuple=True)[0].tolist()
        random.shuffle(indices)

        train_indices.extend(indices[:2])  # first two as train
        val_indices.extend(indices[2:3])  # next as val
        test_indices.extend(indices[3:])  # rest as test

    # data_loder
    batch_size_set = 2
    train_loader = DataLoader(Subset(dataset, train_indices), batch_size=batch_size_set, shuffle=True)
    val_loader = DataLoader(Subset(dataset, val_indices), batch_size=batch_size_set, shuffle=False)
    test_loader = DataLoader(Subset(dataset, test_indices), batch_size=batch_size_set, shuffle=False)

    print(f"Training set size: {len(train_indices)} images")
    print(f"Validation set size: {len(val_indices)} images")
    print(f"Testing set size: {len(test_indices)} images")

    return train_loader, val_loader, test_loader


# # 调用示例
# source_dir = 'C:/Users/LIU/Desktop/face_recog_material/orl_faces'
# train_loader, val_loader, test_loader = prepare_datasets(source_dir)


def resnet101(num_classes=40, pretrained=True):
    if pretrained:
        weights = ResNet101_Weights.DEFAULT
        model = models.resnet101(weights=weights)
    else:
        model = models.resnet101(weights=None)

    # 替换最后的全连接层以适应新的分类任务（40类）
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, num_classes)

    return model


# 实例化模型并检查结构
# model = resnet101()
# print(model)


def train_model(train_loader, val_loader, num_classes, epochs, model_save_path_temp, pretrained=True):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device for training.")

    # 创建模型
    model = resnet101(num_classes=num_classes, pretrained=pretrained)
    model = model.to(device)

    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)  # 每10个epoch学习率降低为原来的0.1

    # 用于绘图的列表
    train_loss_list = []
    val_accuracy_list = []
    best_accuracy = 0.0

    # 训练循环
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        scheduler.step()
        average_loss = running_loss / len(train_loader)
        train_loss_list.append(average_loss)
        # print(f"Epoch {epoch + 1}/{epochs}, Loss: {average_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")

        # 验证
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_accuracy = 100 * correct / total
        val_accuracy_list.append(val_accuracy)
        # print(f'Epoch {epoch + 1}, Validation Accuracy: {val_accuracy:.2f}%')
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {average_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}, Validation Accuracy: {val_accuracy:.2f}%")
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(model.state_dict(), model_save_path_temp)

    # # 绘制结果
    # plt.figure(figsize=(12, 5))
    # plt.subplot(1, 2, 1)
    # plt.plot(range(1, epochs+1), train_loss_list, marker='o', label='Training Loss')
    # plt.title('Training Loss Across Epochs')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.grid(True)
    #
    # plt.subplot(1, 2, 2)
    # plt.plot(range(1, epochs+1), val_accuracy_list, marker='o', color='red', label='Validation Accuracy')
    # plt.title('Validation Accuracy Across Epochs')
    # plt.xlabel('Epoch')
    # plt.ylabel('Accuracy (%)')
    # plt.grid(True)
    #
    # plt.tight_layout()
    # plt.show()

    return model

# # 配置参数
# num_classes = 40  # ORL 数据集类别数
# epochs = 20
#
# model = train_model(train_loader, val_loader, num_classes, epochs)


def test_model(model, test_loader):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} device for testing.")

    model = model.to(device)
    model.eval()  # Set the model to evaluation mode

    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')
    return accuracy

# # Assume that `model` is the trained model from the training step and `test_loader` is ready to use
# test_model(model, test_loader)




# 假设以下函数已经定义好：
# prepare_datasets() - 准备数据集
# train_model() - 训练模型并返回训练后的模型
# test_model() - 测试模型并返回准确率

def run_experiment(source_dir, num_classes, epochs, num_trials):
    accuracies = []
    max_accuracy = 0
    max_accuracy_trial = 0
    model_path_temp = 'D:/Python Project/resnet_model_average_accuracy_storage/model_temp'

    for i in range(num_trials):
        print(f"Running trial {i + 1}/{num_trials}")

        # 准备数据
        train_loader, val_loader, test_loader = prepare_datasets(source_dir)

        # 训练模型
        model = train_model(train_loader, val_loader, num_classes, epochs, model_path_temp)

        # 测试模型
        accuracy = test_model(model, test_loader)
        accuracies.append(accuracy)

        # 更新最高准确率和对应的训练次数
        if accuracy > max_accuracy:
            max_accuracy = accuracy
            max_accuracy_trial = i + 1  # 记录是在哪次训练中达到最高准确率

        # 保存模型
        model_path = f"D:/Python Project/resnet_model_average_accuracy_storage/resnet101_{i + 1}_{accuracy:.2f}%.pth"
        torch.save(model.state_dict(), model_path)

    # 计算平均准确率
    average_accuracy = np.mean(accuracies)
    print(f"Maximum Accuracy was {max_accuracy:.2f}% on trial {max_accuracy_trial}")
    print(f"Average Accuracy over {num_trials} trials: {average_accuracy:.2f}%")

    # 绘制准确率分布图
    plt.hist(accuracies, bins=10, color='blue', alpha=0.7)
    plt.title('Accuracy Distribution')
    plt.xlabel('Accuracy')
    plt.ylabel('Frequency')
    plt.show()

    return accuracies


# 确保已经有一个目录来存储模型
if not os.path.exists('D:/Python Project/resnet_model_average_accuracy_storage'):
    os.makedirs('D:/Python Project/resnet_model_average_accuracy_storage')

# 示例调用
source_dir = 'C:/Users/LIU/Desktop/face_recog_material/orl_faces'
num_classes = 40
epochs = 50
num_trials = 100
accuracies = run_experiment(source_dir, num_classes, epochs, num_trials)




