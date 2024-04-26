from PIL import Image
import torchvision.transforms as transforms
import os
import cv2 as cv
import numpy as np


# Function to augment and save images
def augment_and_save_images(image_path, save_dir='C:/Users/LIU/Desktop/face_recog_material/augmented_data/LHJ', num_images=5):
    # Load the CNN model
    net = cv.dnn.readNetFromCaffe('deploy.prototxt', 'res10_300x300_ssd_iter_140000.caffemodel')

    # # Use CUDA as the preferable backend and target
    # net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
    # net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)

    image_path = image_path  # Replace with the path to your image
    image = cv.imread(image_path)

    # Get the image dimensions
    h, w = image.shape[:2]

    blob = cv.dnn.blobFromImage(cv.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))

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
            # Crop the face area
            face_image = image[startY:endY, startX:endX]
            break

    # Convert the face area from NumPy array to PIL Image
    face_image_pil = Image.fromarray(cv.cvtColor(face_image, cv.COLOR_BGR2RGB))

    augmentations = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.RandomAffine(0, translate=(0.1, 0.1))
    ])

    os.makedirs(save_dir, exist_ok=True)
    for i in range(num_images):
        augmented_image = augmentations(face_image_pil)
        augmented_image.save(f'{save_dir}/augmented_{i}.jpg')
    print("augmentation successful")


# Preprocess and augment the initial face image
augment_and_save_images('C:/Users/LIU/Desktop/face_recog_material/train/lhj1.jpg')




