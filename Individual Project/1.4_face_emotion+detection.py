import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms, utils
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import numpy as np
import random

# Paths to dataset directories
train_dir = 'C:/Users/LIU/Desktop/archive2/train'
test_dir = 'C:/Users/LIU/Desktop/archive2/test'


# # transforms with augmentations for facial emotion recognition
# train_transforms = transforms.Compose([
#     transforms.Grayscale(num_output_channels=1),
#     transforms.Resize((48, 48)),
#     transforms.RandomResizedCrop(48, scale=(0.8, 1.0)),  # Random cropping back to target size
#     transforms.RandomHorizontalFlip(),  # Flipping the image horizontally
#     transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Mild brightness and contrast adjustments
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.5], std=[0.5])
# ])
#
# # Data transforms without augmentation for the test set
# test_transforms = transforms.Compose([
#     transforms.Grayscale(num_output_channels=1),
#     transforms.Resize((48, 48)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.5], std=[0.5])
# ])

# transforms with augmentations for facial emotion recognition
size = 96
train_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((size, size)),
    transforms.RandomResizedCrop(size, scale=(0.8, 1.0)),  # Random cropping back to target size
    transforms.RandomHorizontalFlip(),  # Flipping the image horizontally
    transforms.ColorJitter(brightness=0.2, contrast=0.2),  # Mild brightness and contrast adjustments
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])
test_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((size,size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# Load datasets with augmentation applied to the training set
print("Loading datasets for training...")

train_dataset_full = datasets.ImageFolder(root=train_dir, transform=train_transforms)
test_dataset_full = datasets.ImageFolder(root=test_dir, transform=test_transforms)


# subset of the dataset
def create_subset(dataset):
    subset_size = int(1 * len(dataset))  # 10% of the dataset
    indices = list(range(len(dataset)))
    # Ensure subset size is a multiple of the batch size
    subset_size -= subset_size % 64
    subset_indices = random.sample(indices, subset_size)
    return Subset(dataset, subset_indices)


train_dataset = create_subset(train_dataset_full)
test_dataset = create_subset(test_dataset_full)

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
print("Datasets loaded successfully.")


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


# Initialize model, loss function, and optimizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = EmotionCNN().to(device)

num_classes = len(train_dataset_full.classes)  # num_classes = 7
model = EmotionCNN(num_classes=num_classes).to(device)
# print(train_dataset_full.classes)  # ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)  # decays the learning rate by a factor gamma
# Training loop
epochs = 200
losses = []
for epoch in range(epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)  # Performs a forward pass through the model and getting predictions
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    epoch_loss = running_loss / len(train_loader.dataset)
    losses.append(epoch_loss)  # Store the epoch loss
    scheduler.step()  # Update the learning rate
    print(f'Epoch {epoch + 1}/{epochs}, Loss: {running_loss / len(train_loader.dataset):.4f}, LR: {scheduler.get_last_lr()[0]:.6f}')

plt.figure(figsize=(10, 6))
plt.plot(range(1, epochs+1), losses, marker='o', linestyle='-', color='blue')
plt.title('Training Loss vs. Epoch')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.xticks(range(1, epochs+1))
plt.show()


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


# Function to display test images with their predicted emotions
def display_predictions(loader, num_images):
    images, labels = next(iter(loader))
    images, labels = images.to(device), labels.to(device)
    outputs = model(images)
    _, preds = torch.max(outputs, 1)

    plt.figure(figsize=(25, 5))
    for i in range(num_images):
        ax = plt.subplot(1, num_images, i + 1)
        img = images[i].cpu().numpy().transpose((1, 2, 0))
        img = np.clip((img * 0.5 + 0.5), 0, 1)  # Unnormalize
        plt.imshow(img.squeeze(), cmap='gray')
        ax.set_title(f'Predicted: {test_dataset_full.classes[preds[i]]}\nActual: {test_dataset_full.classes[labels[i]]}')
        plt.axis('off')
    plt.show()


# Calculate and print the accuracy
accuracy = calculate_accuracy(model, test_loader)
print(f'Accuracy on test set: {accuracy:.2f}%')

# Display 10 test images with their predicted emotions
display_predictions(test_loader, num_images=5)

torch.save(model.state_dict(), 'C:/Users/LIU/Desktop/face_detection_model/face_emotion_model6')
