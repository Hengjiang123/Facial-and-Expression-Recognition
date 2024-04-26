import cv2 as cv
import os
import numpy as np


def prepare_training_data(data_folder_path):
    # Get the directory of the data folder
    dirs = os.listdir(data_folder_path)

    # Initialising faces and tag lists
    faces = []
    labels = []

    # Iterate through each directory and read the images
    for dir_name in dirs:
        if not dir_name.startswith("s"):
            continue

        label = int(dir_name.replace("s", ""))
        subject_dir_path = os.path.join(data_folder_path, dir_name)

        subject_images_names = os.listdir(subject_dir_path)

        for image_name in subject_images_names:
            if image_name.startswith("."):
                continue

            image_path = os.path.join(subject_dir_path, image_name)
            image = cv.imread(image_path, cv.IMREAD_GRAYSCALE)

            # resize the image
            desired_size = (200, 200)
            image = cv.resize(image, desired_size)

            if image is None:
                continue

            # Adding read images and tags to the list
            faces.append(image)
            labels.append(label)

    return faces, labels


# Train and save the model
def train_and_save_model(data_folder_path, model_save_path):
    print("Preparing data...")
    faces, labels = prepare_training_data(data_folder_path)
    print(f"Data prepared. Faces: {len(faces)}, Labels: {len(labels)}")

    # Create and train LBPH face recogniser
    face_recognizer = cv.face.LBPHFaceRecognizer_create()
    # face_recognizer = cv.face.EigenFaceRecognizer_create()
    # face_recognizer = cv.face.FisherFaceRecognizer_create()

    face_recognizer.train(faces, np.array(labels))  # training here!

    # save the trained model
    face_recognizer.save(model_save_path)
    print(f"Model trained and saved at {model_save_path}")


# direction
data_folder_path = 'C:/Users/LIU/Desktop/face_recog_material/datasets'
model_save_path = 'C:/Users/LIU/Desktop/face_recog_material/model/trained_model_LBPH.xml'

train_and_save_model(data_folder_path, model_save_path)

