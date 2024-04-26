import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import cv2 as cv
import numpy as np
import torchvision.models as models


Exp_model_path = 'C:/Users/LIU/Desktop/face_detection_model/face_emotion_model5_73%'


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


def initialize_resnet101():
    # Load a pre-trained ResNet-101 model
    model = models.resnet101(weights=None)

    # Assuming the model is trained for 1000 classes, and you want to adapt it for fewer classes
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 40)  # Assuming 100 different identities

    # Load your custom trained weights (adjust the path as needed)
    model.load_state_dict(torch.load('C:/Users/LIU/Desktop/ResNet_LHJ_model/ORL_LHJ_1_100.00%.pth'))

    model.to(device)
    model.eval()
    return model


# Initialize the exp and face model and load the saved state dict
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
num_classes = 7
Exp_model = EmotionCNN(num_classes=num_classes).to(device)
Exp_model.load_state_dict(torch.load(Exp_model_path))

# Move the exp model to the appropriate device (GPU or CPU)
Exp_model.to(device)
Exp_model.eval()  # Set the model to evaluation mode

# set the face model
face_model = initialize_resnet101()


def detect_emotion_face_with_gpu():
    # Load the CNN model
    net = cv.dnn.readNetFromCaffe('deploy.prototxt', 'res10_300x300_ssd_iter_140000.caffemodel')

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

    # 人脸标签
    person_names = ['10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '2', '20', '21', '22', '23', '24',
                    '25', '26', '27', '28', '29', '3', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39',
                    '4', '40', '5', '6', '7', '8', '9', 'Hengjiang']
    # directory = 'C:/Users/LIU/Desktop/face_recog_material/orl_faces_LHJ'
    # # Loop over the contents of the directory
    # for name in os.listdir(directory):
    #     # Check if the item is a directory and starts with 's'
    #     if os.path.isdir(os.path.join(directory, name)) and name.startswith('s'):
    #         # Remove the 's' to get the person's name and add to list
    #         person_names.append(name[1:])  # Removes the first character 's'
    # print(person_names)

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

                # Resize the face ROI to the input size expected by model (e.g., 48x48), then preprocess it
                face = cv.resize(face, (96, 96))
                face = face / 255.0  # Normalization
                face = np.expand_dims(face, axis=0)
                face = np.expand_dims(face, axis=0)  # Add batch dimension
                face = torch.from_numpy(face).type(torch.FloatTensor).to(device)

                # Make the exp prediction
                outputs_exp = Exp_model(face)
                _, preds = torch.max(outputs_exp, 1)
                emotion = emotions[preds.item()]

                face_f = frame[startY:endY, startX:endX]
                face_f_rgb = cv.cvtColor(face_f, cv.COLOR_BGR2RGB)  # Convert BGR to RGB
                face_f_tensor = preprocess(face_f_rgb)  # Apply transformations
                face_f_tensor = face_f_tensor.unsqueeze(0).to(device)  # Add a batch dimension and send to device

                # Make the face prediction
                outputs_face = face_model(face_f_tensor)
                _, predicted_id = torch.max(outputs_face, 1)
                face_ID = person_names[predicted_id.item()]  # id_to_name maps ID to person's name

                # Draw the bounding box of the face along with the associated probability
                cv.rectangle(frame, (startX, startY), (endX, endY), (255, 0, 0), 2)
                label = "{} - {}".format(face_ID, emotion)
                cv.putText(frame, label, (startX, startY - 10), cv.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)

        # Display the resulting frame
        cv.imshow('Face Detection', frame)

        # Break the loop on 'q' key press
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture
    cap.release()
    cv.destroyAllWindows()


detect_emotion_face_with_gpu()

