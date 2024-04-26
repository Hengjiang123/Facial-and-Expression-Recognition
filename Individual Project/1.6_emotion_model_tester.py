import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms, utils
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import numpy as np
import random

# model_path = 'C:/Users/LIU/Desktop/face_detection_model/face_emotion_model3'
# test_dir = 'C:/Users/LIU/Desktop/archive/test'
#
# # Define the same transforms used during training (excluding augmentation)
# test_transforms = transforms.Compose([
#     transforms.Grayscale(num_output_channels=1),
#     transforms.Resize((48, 48)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.5], std=[0.5])
# ])
#
# test_dataset_full = datasets.ImageFolder(root=test_dir, transform=test_transforms)
#
#
# def create_subset(dataset):
#     subset_size = int(1 * len(dataset))  # 10% of the dataset
#     indices = list(range(len(dataset)))
#     subset_size -= subset_size % 64  # Ensure subset size is a multiple of the batch size
#     subset_indices = random.sample(indices, subset_size)
#     return Subset(dataset, subset_indices)
#
#
# # Load the test dataset
# test_dataset = create_subset(test_dataset_full)
# test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
#
#
# # Define the same model architecture used for training
# class EmotionCNN(nn.Module):
#     def __init__(self, num_classes):
#         super(EmotionCNN, self).__init__()
#         self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
#         self.bn1 = nn.BatchNorm2d(32)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
#         self.bn2 = nn.BatchNorm2d(64)
#         self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
#         self.bn3 = nn.BatchNorm2d(128)
#
#         self.pool = nn.MaxPool2d(2, 2)
#         self.dropout = nn.Dropout(0.5)
#         self.fc1 = nn.Linear(6 * 6 * 128, 1024)  # Adjusted for additional conv layer
#         self.fc2 = nn.Linear(1024, num_classes)
#
#     def forward(self, x):
#         x = F.relu(self.bn1(self.conv1(x)))
#         x = self.pool(x)
#         x = F.relu(self.bn2(self.conv2(x)))
#         x = self.pool(x)
#         x = F.relu(self.bn3(self.conv3(x)))
#         x = self.pool(x)
#
#         x = torch.flatten(x, 1)
#         x = self.dropout(x)
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x
#
#
# # Initialize the model and load the saved state dict
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# num_classes = len(test_dataset_full.classes)  # Assuming train_dataset_full is available in your context
# model = EmotionCNN(num_classes=num_classes).to(device)
# model.load_state_dict(torch.load(model_path))
#
# # Move the model to the appropriate device (GPU or CPU)
# model.to(device)
# model.eval()  # Set the model to evaluation mode
#
#
# # Function to calculate accuracy
# def calculate_accuracy(model, loader):
#     correct = 0
#     total = 0
#     with torch.no_grad():
#         for images, labels in loader:
#             images, labels = images.to(device), labels.to(device)
#             outputs = model(images)
#             _, predicted = torch.max(outputs.data, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
#     return 100 * correct / total
#
#
# # Calculate and print the accuracy
# accuracy = calculate_accuracy(model, test_loader)
# print(f'Accuracy on test set: {accuracy:.2f}%')
#
#
# # Function to display images with actual and predicted labels
# def display_images_with_predictions(model, loader, num_images=5):
#     images, labels = next(iter(loader))
#     images, labels = images.to(device), labels.to(device)
#     outputs = model(images)
#     _, preds = torch.max(outputs, 1)
#
#     fig, axes = plt.subplots(1, num_images, figsize=(15, 3))
#     for i in range(num_images):
#         idx = random.randint(0, len(images) - 1)  # Pick a random image
#         ax = axes[i]
#         img = images[idx].cpu().numpy().transpose((1, 2, 0))
#         img = np.clip((img * 0.5 + 0.5), 0, 1)  # Unnormalize
#         ax.imshow(img.squeeze(), cmap='gray')
#         ax.set_title(f'Predicted: {test_dataset_full.classes[preds[idx]]}\nActual: {test_dataset_full.classes[labels[idx]]}')
#         ax.axis('off')
#     plt.show()
#
#
# # Display 5 random test images with predictions
# display_images_with_predictions(model, test_loader, num_images=5)




















model_path = 'C:/Users/LIU/Desktop/face_detection_model/face_emotion_model5'
test_dir = 'C:/Users/LIU/Desktop/archive1/test'

# Define the same transforms used during training (excluding augmentation)
size = 96
test_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((size, size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

test_dataset_full = datasets.ImageFolder(root=test_dir, transform=test_transforms)


def create_subset(dataset):
    subset_size = int(0.1 * len(dataset))  # 10% of the dataset
    indices = list(range(len(dataset)))
    subset_size -= subset_size % 64  # Ensure subset size is a multiple of the batch size
    subset_indices = random.sample(indices, subset_size)
    return Subset(dataset, subset_indices)


# Load the test dataset
test_dataset = create_subset(test_dataset_full)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


# Define the same model architecture used for training
class EmotionCNN(nn.Module):
    def __init__(self, num_classes):
        super(EmotionCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(512)
        self.conv6 = nn.Conv2d(512, 512, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(6 * 6 * 512, 512)  # Adjusted for additional conv layer
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool(x)
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn5(self.conv6(x)))
        x = F.relu(self.bn5(self.conv6(x)))

        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Initialize the model and load the saved state dict
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = len(test_dataset_full.classes)  # Assuming train_dataset_full is available in your context
model = EmotionCNN(num_classes=num_classes).to(device)
model.load_state_dict(torch.load(model_path))

# Move the model to the appropriate device (GPU or CPU)
model.to(device)
model.eval()  # Set the model to evaluation mode


# Function to calculate accuracy
def calculate_accuracy(model, loader):
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return 100 * correct / total


# Calculate and print the accuracy
accuracy = calculate_accuracy(model, test_loader)
print(f'Accuracy on test set: {accuracy:.2f}%')


# Function to display images with actual and predicted labels
def display_images_with_predictions(model, loader, num_images=5):
    images, labels = next(iter(loader))
    images, labels = images.to(device), labels.to(device)
    outputs = model(images)
    _, preds = torch.max(outputs, 1)

    fig, axes = plt.subplots(1, num_images, figsize=(15, 3))
    for i in range(num_images):
        idx = random.randint(0, len(images) - 1)  # Pick a random image
        ax = axes[i]
        img = images[idx].cpu().numpy().transpose((1, 2, 0))
        img = np.clip((img * 0.5 + 0.5), 0, 1)  # Unnormalize
        ax.imshow(img.squeeze(), cmap='gray')
        ax.set_title(f'Predicted: {test_dataset_full.classes[preds[idx]]}\nActual: {test_dataset_full.classes[labels[idx]]}')
        ax.axis('off')
    plt.show()


# Display 5 random test images with predictions
display_images_with_predictions(model, test_loader, num_images=5)

