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
import cv2 as cv
import numpy as np


model_path = 'C:/Users/LIU/Desktop/face_detection_model/face_emotion_model5'


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
num_classes = 7
model = EmotionCNN(num_classes=num_classes).to(device)
model.load_state_dict(torch.load(model_path))

# Move the model to the appropriate device (GPU or CPU)
model.to(device)
model.eval()  # Set the model to evaluation mode



def detect_emotion_with_gpu():
    # Load the CNN model
    net = cv.dnn.readNetFromCaffe('deploy.prototxt', 'res10_300x300_ssd_iter_140000.caffemodel')

    # # Use CUDA as the preferable backend and target
    # net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
    # net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)

    # Initialize video capture
    cap = cv.VideoCapture(0)

    # 定义转换操作，以匹配模型的输入要求
    preprocess = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((96, 96)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 表情标签
    emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the captured frame to grayscale to match the input expectation of the model
        gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # Get the frame dimensions
        (h, w) = frame.shape[:2]

        # Preprocess the frame for the neural network
        blob = cv.dnn.blobFromImage(cv.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

        # Pass the blob through the network and obtain the detections
        net.setInput(blob)
        detections = net.forward()

        # Loop over the detections
        for i in range(0, detections.shape[2]):
            # Extract the confidence of the detection
            confidence = detections[0, 0, i, 2]

            # Filter out weak detections by ensuring the confidence is greater than a minimum threshold
            if confidence > 0.5:
                # Compute the coordinates of the bounding box for the face
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # Ensure the bounding box fits within the frame dimensions
                startX, startY, endX, endY = max(0, startX), max(0, startY), min(w - 1, endX), min(h - 1, endY)

                # Extract the face ROI
                face = gray_frame[startY:endY, startX:endX]

                # Resize the face ROI to the input size expected by our model (e.g., 48x48), then preprocess it
                face = cv.resize(face, (96, 96))
                face = face / 255.0  # Normalization
                face = np.expand_dims(face, axis=0)
                face = np.expand_dims(face, axis=0)  # Add batch dimension
                face = torch.from_numpy(face).type(torch.FloatTensor).to(device)

                # Make the prediction
                outputs = model(face)
                _, preds = torch.max(outputs, 1)
                emotion = emotions[preds.item()]

                # Draw the bounding box of the face along with the associated probability
                cv.rectangle(frame, (startX, startY), (endX, endY), (255, 0, 0), 2)
                label = "{:.2f}% - {}".format(confidence * 100, emotion)
                cv.putText(frame, label, (startX, startY - 10), cv.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

        # Display the resulting frame
        cv.imshow('Face Detection', frame)

        # Break the loop on 'q' key press
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture
    cap.release()
    cv.destroyAllWindows()


detect_emotion_with_gpu()

