import cv2 as cv
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import os
import numpy as np
from PIL import Image
import random
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F



# # LBPH+orl
# def augment_image(image):
#     augmented_images = []
#
#     # Flip the image horizontally
#     flipped = cv.flip(image, 1)
#     augmented_images.append(flipped)
#
#     # Rotate the image slightly left and right
#     rows, cols = image.shape
#     M_right = cv.getRotationMatrix2D((cols / 2, rows / 2), -5, 1)
#     M_left = cv.getRotationMatrix2D((cols / 2, rows / 2), 5, 1)
#     rotated_right = cv.warpAffine(image, M_right, (cols, rows))
#     rotated_left = cv.warpAffine(image, M_left, (cols, rows))
#
#     augmented_images.extend([rotated_right, rotated_left])
#
#     # Return the original and augmented images
#     augmented_images.append(image)
#     return augmented_images
#
#
# def data_prepare(data_folder_path):
#     dirs = os.listdir(data_folder_path)
#     faces_train = []
#     labels_train = []
#     faces_test = []
#     labels_test = []
#
#     for dir_name in dirs:
#         if not dir_name.startswith("s"):
#             continue
#
#         label = int(dir_name.replace("s", ""))
#         subject_dir_path = os.path.join(data_folder_path, dir_name)
#         subject_images_names = os.listdir(subject_dir_path)
#
#         # Randomly select 9 pictures for training, the rest for testing
#         random.shuffle(subject_images_names)
#         training_images_names = subject_images_names[:1]
#         testing_images_names = subject_images_names[1:]
#
#         for image_name in training_images_names:
#             if image_name.startswith("."):
#                 continue
#
#             image_path = os.path.join(subject_dir_path, image_name)
#             image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
#             image = cv.resize(image, (200, 200))
#
#             if image is None:
#                 continue
#
#             # Augment each training image
#             augmented_images = augment_image(image)
#             for aug_image in augmented_images:
#                 faces_train.append(aug_image)
#                 labels_train.append(label)
#
#         for image_name in testing_images_names:
#             if image_name.startswith("."):
#                 continue
#
#             image_path = os.path.join(subject_dir_path, image_name)
#             image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
#             image = cv.resize(image, (200, 200))
#
#             if image is None:
#                 continue
#
#             faces_test.append(image)
#             labels_test.append(label)
#
#     return faces_train, labels_train, faces_test, labels_test
#
#
# def train_and_evaluate_model(data_folder_path, model_save_path):
#     print("Preparing data...")
#     faces_train, labels_train, faces_test, labels_test = data_prepare(data_folder_path)
#     print(f"Data prepared. Training Faces: {len(faces_train)}, Labels: {len(labels_train)}")
#
#     face_recognizer = cv.face.LBPHFaceRecognizer_create()
#     # face_recognizer = cv.face.FisherFaceRecognizer_create()
#     # face_recognizer = cv.face.EigenFaceRecognizer_create()
#     face_recognizer.train(faces_train, np.array(labels_train))
#     # face_recognizer.save(model_save_path)
#     print(f"LBPH Model trained and saved at {model_save_path}")
#
#     # test model
#     correct_predictions = 0
#     for face, label in zip(faces_test, labels_test):
#         predicted_label, confidence = face_recognizer.predict(face)
#         if predicted_label == label:
#             correct_predictions += 1
#
#     accuracy = correct_predictions / len(faces_test)
#     print(f"Model accuracy: {accuracy * 100:.2f}%")
#
#
# data_folder_path = 'C:/Users/LIU/Desktop/face_recog_material/orl_faces'
# model_save_path = 'C:/Users/LIU/Desktop/face_recog_material/model/trained_model_LBPH_orl.xml'
#
# train_and_evaluate_model(data_folder_path, model_save_path)
#
#
#
#





# # LBPH+CASIA
# def load_images_from_folder(folder_path, label, img_size=(200, 200)):
#     images = []
#     labels = []
#
#     for image_name in os.listdir(folder_path):
#         image_path = os.path.join(folder_path, image_name)
#         image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)
#         if image is not None:
#             image = cv.resize(image, img_size)  # Resize the image
#             images.append(image)
#             labels.append(label)
#
#     return images, labels
#
# def prepare_data(train_data_path, test_data_path, num_classes=1000, img_size=(200, 200)):
#     train_faces = []
#     train_labels = []
#     test_faces = []
#     test_labels = []
#
#     # Load training images
#     for i in range(1, num_classes + 1):
#         dir_name = f"s{i}"
#         label = i - 1  # Adjust label to be 0-indexed
#         train_folder_path = os.path.join(train_data_path, dir_name)
#         test_folder_path = os.path.join(test_data_path, dir_name)
#
#         # Load training images
#         train_images, train_labels_for_dir = load_images_from_folder(train_folder_path, label, img_size)
#         train_faces.extend(train_images)
#         train_labels.extend(train_labels_for_dir)
#
#         # Load testing images
#         test_images, test_labels_for_dir = load_images_from_folder(test_folder_path, label, img_size)
#         test_faces.extend(test_images)
#         test_labels.extend(test_labels_for_dir)
#
#     return np.array(train_faces), np.array(train_labels), np.array(test_faces), np.array(test_labels)
#
#
# def train_and_test_model(train_data_path, test_data_path, model_save_path):
#     print("Preparing data...")
#     faces, labels, test_faces, test_labels = prepare_data(train_data_path, test_data_path)
#     print(f"Data prepared. Training Faces: {len(faces)}, Test Faces: {len(test_faces)}")
#
#     # Train LBPH face recognizer
#     face_recognizer = cv.face.LBPHFaceRecognizer_create()
#     # face_recognizer = cv.face.FisherFaceRecognizer_create()
#     # face_recognizer = cv.face.EigenFaceRecognizer_create()
#     face_recognizer.train(faces, np.array(labels))
#     face_recognizer.save(model_save_path)
#     print("Model trained and saved.")
#
#     # Test the model
#     correct_predictions = 0
#     for i in range(len(test_faces)):
#         prediction, _ = face_recognizer.predict(test_faces[i])
#         if prediction == test_labels[i]:
#             correct_predictions += 1
#
#     accuracy = correct_predictions / len(test_faces)
#     print(f"Test Accuracy: {accuracy * 100:.2f}%")
#
#
# train_data_path = 'C:/Users/LIU/Desktop/face_recog_material/CASIA/train'
# test_data_path = 'C:/Users/LIU/Desktop/face_recog_material/CASIA/test'
# model_save_path = 'C:/Users/LIU/Desktop/face_recog_material/model/trained_model_LBPH_CASIA.xml'
#
# train_and_test_model(train_data_path, test_data_path, model_save_path)












# CNN+orl
class FaceDataset(Dataset):
    def __init__(self, data_folder_path, transform=None, train=True, augmentations=10):
        self.data_folder_path = data_folder_path
        self.transform = transform
        self.train = train
        self.augmentations = augmentations if train else 1
        self.faces, self.labels = self.prepare_data()

    def prepare_data(self):
        faces = []
        labels = []
        dirs = os.listdir(self.data_folder_path)
        dirs = [d for d in dirs if d.startswith("s")]

        for dir_name in dirs:
            label = int(dir_name.replace("s", "")) - 1
            subject_dir_path = os.path.join(self.data_folder_path, dir_name)
            subject_images_names = sorted(os.listdir(subject_dir_path))

            if self.train:
                # Select specified number of images for training and augment them
                selected_images = subject_images_names[:9]
                augmented_images = selected_images * 10  # Repeat each selected image 10 times for augmentation
            else:
                # Use remaining images for testing, no augmentation
                selected_images = subject_images_names[9:]
                augmented_images = selected_images

            for image_name in augmented_images:
                if image_name.startswith("."):
                    continue

                image_path = os.path.join(subject_dir_path, image_name)
                faces.append(image_path)
                labels.append(label)

        print(f"Faces: {len(faces)}, Labels: {len(labels)}")
        return faces, labels

    def __len__(self):
        return len(self.faces)

    def __getitem__(self, idx):
        image_path = self.faces[idx]
        image = Image.open(image_path).convert("L")

        # Define the augmentation transformations
        augmentation_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.Resize((200, 200)),
            transforms.ToTensor(),
        ])

        if self.train:
            # Apply augmentations for the training images
            image = augmentation_transform(image)
        elif self.transform:
            # Apply the default transform for the test images
            image = self.transform(image)

        label = self.labels[idx]
        return image, label



# CNN model
class CNN_orl(nn.Module):
    def __init__(self, num_classes):
        super(CNN_orl, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5, stride=1, padding=2)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2)
        self.fc1 = nn.Linear(64 * 50 * 50, 1000)
        self.fc2 = nn.Linear(1000, num_classes)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 50 * 50)  # Flatten
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


def train_and_evaluate_model(data_folder_path, num_classes):
    transform = transforms.Compose([
        transforms.Resize((200, 200)),
        transforms.ToTensor(),
    ])

    # training dataset
    print("train data:")
    train_dataset = FaceDataset(data_folder_path, transform=transform, train=True, augmentations=10)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

    # test dataset
    print("test data:")
    test_dataset = FaceDataset(data_folder_path, transform=transform, train=False)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    print(f"Training batch: {len(train_loader)}, Test batch: {len(test_loader)}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CNN_orl(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # learning rate scheduler
    scheduler = StepLR(optimizer, step_size=1, gamma=0.6)

    # train the model
    for epoch in range(6):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data[0].to(device), data[1].to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 10 == 9:  # print every 10 mini-batches
                print(f"[{epoch + 1}, {i + 1}] loss: {running_loss / 10:.3f}")
                running_loss = 0.0
        scheduler.step()  # update learning rate
        current_lr = scheduler.get_last_lr()[0]
        print(f'Epoch {epoch + 1}, Current Learning Rate: {current_lr}')

        # test the model
        correct_predictions = 0
        total_predictions = 0
        with torch.no_grad():
            for data in test_loader:
                inputs, labels = data[0].to(device), data[1].to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total_predictions += labels.size(0)
                correct_predictions += (predicted == labels).sum().item()

        accuracy = correct_predictions / total_predictions
        print(f'Model accuracy on test set: {accuracy * 100:.2f}%')


# 设置数据路径和类别数
data_folder_path = 'C:/Users/LIU/Desktop/face_recog_material/orl_faces'


train_and_evaluate_model(data_folder_path, num_classes = 40)








# # CNN_CASIA
# class CNN_CASIA(nn.Module):
#     def __init__(self, num_classes=100):
#         super(CNN_CASIA, self).__init__()
#         self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
#         self.bn1 = nn.BatchNorm2d(32)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
#         self.bn2 = nn.BatchNorm2d(64)
#         self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
#         self.bn3 = nn.BatchNorm2d(128)
#         self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
#         self.bn4 = nn.BatchNorm2d(256)
#         self.conv5 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
#         self.bn5 = nn.BatchNorm2d(512)
#         self.conv6 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
#         self.bn6 = nn.BatchNorm2d(512)
#
#         self.pool = nn.MaxPool2d(2, 2)
#         self.dropout = nn.Dropout(0.5)
#         # Adjusted for 112x112 input image size
#         self.fc1 = nn.Linear(6 * 6 * 512, 512)
#         self.fc2 = nn.Linear(512, num_classes)
#
#     def forward(self, x):
#         x = self.pool(F.relu(self.bn1(self.conv1(x))))
#         x = self.pool(F.relu(self.bn2(self.conv2(x))))
#         x = self.pool(F.relu(self.bn3(self.conv3(x))))
#         x = self.pool(F.relu(self.bn4(self.conv4(x))))
#         x = self.pool(F.relu(self.bn5(self.conv5(x))))
#         x = self.pool(F.relu(self.bn6(self.conv6(x))))
#
#         x = torch.flatten(x, 1)
#         x = self.dropout(x)
#         x = F.relu(self.fc1(x))
#         x = self.fc2(x)
#         return x
#
#
# class CASIADataset(Dataset):
#     def __init__(self, root_dir, train=True, transform=None):
#         self.root_dir = root_dir
#         self.transform = transform
#         self.train = train
#         self.samples = []
#
#         folder_name = 'train' if self.train else 'test'
#         self.root_dir = os.path.join(self.root_dir, folder_name)
#
#         for label in range(1, 101):  # first 100 label
#             folder_path = os.path.join(self.root_dir, f's{label}')
#             for img_name in os.listdir(folder_path)[:20]:  # choose 20 pictures from each label
#                 self.samples.append((os.path.join(folder_path, img_name), label - 1))
#
#     def __len__(self):
#         return len(self.samples)
#
#     def __getitem__(self, idx):
#         img_path, label = self.samples[idx]
#         image = Image.open(img_path).convert('L')  # turn to grayscale
#         if self.transform:
#             image = self.transform(image)
#         return image, label
#
#
#
# def train_test_model(data_folder_path, num_classes=100, num_epochs=10, batch_size=32):
#     transform_train = transforms.Compose([
#         transforms.RandomHorizontalFlip(),
#         transforms.RandomRotation(10),
#         transforms.Resize((112, 112)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485], std=[0.229]),
#     ])
#
#     transform_test = transforms.Compose([
#         transforms.Resize((112, 112)),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=[0.485], std=[0.229]),
#     ])
#
#     train_dataset = CASIADataset(root_dir=data_folder_path, train=True, transform=transform_train)
#     test_dataset = CASIADataset(root_dir=data_folder_path, train=False, transform=transform_test)
#
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#     test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
#
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     model = CNN_CASIA(num_classes=num_classes).to(device)
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
#
#     # learning rate scheduler
#     scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
#
#     # train model
#     model.train()
#     for epoch in range(num_epochs):
#         running_loss = 0.0
#         for images, labels in train_loader:
#             images, labels = images.to(device), labels.to(device)
#
#             optimizer.zero_grad()
#
#             outputs = model(images)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()
#
#             running_loss += loss.item()
#
#         scheduler.step()  # update learning rate
#
#         print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}')
#
#     # test model
#     model.eval()
#     correct = 0
#     total = 0
#     with torch.no_grad():
#         for images, labels in test_loader:
#             images, labels = images.to(device), labels.to(device)
#             outputs = model(images)
#             _, predicted = torch.max(outputs.data, 1)
#             total += labels.size(0)
#             correct += (predicted == labels).sum().item()
#
#     print(f'Accuracy on test set: {100 * correct / total}%')
#
#
# data_folder_path = 'C:/Users/LIU/Desktop/face_recog_material/CASIA'
# train_test_model(data_folder_path)
#
# # train_data_path = 'C:/Users/LIU/Desktop/face_recog_material/CASIA/train'
# # test_data_path = 'C:/Users/LIU/Desktop/face_recog_material/CASIA/test'











# data_folder_path = 'C:/Users/LIU/Desktop/CASIA_1000/casia-webface'
# output_folder_path = 'C:/Users/LIU/Desktop/face_recog_material/CASIA/test'
# num_images_per_person = 5
#
# if not os.path.exists(output_folder_path):
#     os.makedirs(output_folder_path)
#
# # 数据增强
# transform = transforms.Compose([
#     transforms.RandomHorizontalFlip(),
#     transforms.ColorJitter(),
#     transforms.RandomRotation(20),
# ])
#
# dirs = sorted(os.listdir(data_folder_path))[:1000]  # 假设只处理前1000个人
#
# for label, dir_name in enumerate(dirs):
#     subject_dir_path = os.path.join(data_folder_path, dir_name)
#     output_dir_path = os.path.join(output_folder_path, f's{label}')
#     if not os.path.exists(output_dir_path):
#         os.makedirs(output_dir_path)
#
#     subject_images_names = os.listdir(subject_dir_path)
#     selected_images = random.sample(subject_images_names, min(len(subject_images_names), num_images_per_person))
#
#     # 复制或增强图片
#     for i, image_name in enumerate(selected_images):
#         image_path = os.path.join(subject_dir_path, image_name)
#         output_image_path = os.path.join(output_dir_path, f'{i + 1}.jpg')
#
#         image = Image.open(image_path)
#         image.save(output_image_path)
#
#     # 数据增强补齐不足的图片
#     for i in range(len(selected_images), num_images_per_person):
#         image_name = random.choice(subject_images_names)  # 从选中的图片中随机选一张进行增强
#         image_path = os.path.join(subject_dir_path, image_name)
#         output_image_path = os.path.join(output_dir_path, f'{i + 1}.jpg')
#
#         image = Image.open(image_path).convert("RGB")
#         transformed_image = transform(image)
#         transformed_image.save(output_image_path)
#
#     # 删除多余的图片
#     for image_name in subject_images_names:
#         if image_name not in selected_images:
#             image_path = os.path.join(subject_dir_path, image_name)
#             try:
#                 os.remove(image_path)
#             except OSError:
#                 pass
#





