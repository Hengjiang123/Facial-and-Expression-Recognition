import matplotlib.pyplot as plt
from facenet_pytorch import MTCNN
import torch
from PIL import Image
import numpy as np


def align_face(image, mtcnn):
    # image = Image.open(image_path)
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # Detect faces and landmarks
    boxes, probs, landmarks = mtcnn.detect(image, landmarks=True)

    if landmarks is not None and len(landmarks) > 0:
        # Assuming the first face and landmarks in the image
        eye_left = np.array(landmarks[0][0])
        eye_right = np.array(landmarks[0][1])

        # Calculate rotation angle to make the eyes horizontal
        eye_direction = eye_right - eye_left
        rotation = np.degrees(np.arctan2(eye_direction[1], eye_direction[0]))

        # Calculate the center between the two eyes for the rotation
        center_of_eyes = ((eye_left[0] + eye_right[0]) / 2, (eye_left[1] + eye_right[1]) / 2)

        # Rotate image
        rotated_image = image.rotate(rotation, center=center_of_eyes, resample=Image.BICUBIC)

        # Calculate the size of the face
        eye_distance = np.linalg.norm(eye_left - eye_right)
        # Assuming the face width is approximately 2x of the distance between the eyes
        face_width = 2.5 * eye_distance
        # Assuming the face height is approximately 2.5x of the distance between the eyes
        face_height = 2.8 * eye_distance

        # Define a box to crop the face, centered around the mid-point between the eyes
        # Adjust these as necessary to capture the full face
        left = center_of_eyes[0] - face_width // 2
        top = center_of_eyes[1] - face_height // 2.5
        right = center_of_eyes[0] + face_width // 2
        bottom = center_of_eyes[1] + 1.5 * face_height // 2.5

        # Crop the face
        cropped_face = rotated_image.crop((left, top, right, bottom))

        plt.imshow(cropped_face)
        plt.show()

        return cropped_face
    else:
        # If no face is detected, return the original image
        return image


mtcnn = MTCNN(keep_all=True, select_largest=True, post_process=False, device=torch.device('cpu'))
image = Image.open('C:/Users/LIU/Desktop/face-iv/lhj3.jpg')
aligned_image = align_face(image, mtcnn)





# 标出landmark:

# import matplotlib.pyplot as plt
# from facenet_pytorch import MTCNN
# import torch
# from PIL import Image, ImageDraw
# import numpy as np
#
# def align_and_show_face(image_path, mtcnn):
#     image = Image.open(image_path)
#     if image.mode != 'RGB':
#         image = image.convert('RGB')
#
#     draw = ImageDraw.Draw(image)  # To draw on the image
#
#     # Detect faces and landmarks
#     boxes, probs, landmarks = mtcnn.detect(image, landmarks=True)
#
#     # Draw landmarks on the original image
#     if landmarks is not None:
#         for face_landmarks in landmarks:
#             for (x, y) in face_landmarks:
#                 draw.ellipse([(x - 10, y - 10), (x + 10, y + 10)], fill='red')
#
#     if landmarks is not None and len(landmarks) > 0:
#         # Assuming the first face and landmarks in the image
#         eye_left = np.array(landmarks[0][0])
#         eye_right = np.array(landmarks[0][1])
#
#         # Calculate rotation angle to make the eyes horizontal
#         eye_direction = eye_right - eye_left
#         rotation = np.degrees(np.arctan2(eye_direction[1], eye_direction[0]))
#
#         # Calculate the center between the two eyes for the rotation
#         center_of_eyes = ((eye_left[0] + eye_right[0]) / 2, (eye_left[1] + eye_right[1]) / 2)
#
#         # Rotate image
#         rotated_image = image.rotate(rotation, center=center_of_eyes, resample=Image.BICUBIC)
#
#         # Calculate the size of the face
#         eye_distance = np.linalg.norm(eye_left - eye_right)
#         face_width = 2.5 * eye_distance  # Assuming the face width is approximately 2x of the distance between the eyes
#         face_height = 2.8 * eye_distance  # Assuming the face height is approximately 2.5x of the distance between the eyes
#
#         # Define a box to crop the face, centered around the mid-point between the eyes
#         left = center_of_eyes[0] - face_width // 2
#         top = center_of_eyes[1] - face_height // 2.5
#         right = center_of_eyes[0] + face_width // 2
#         bottom = center_of_eyes[1] + 1.5 * face_height // 2.5
#
#         # Crop the face
#         cropped_face = rotated_image.crop((left, top, right, bottom))
#
#         plt.figure(figsize=(10, 5))
#
#         # Show original image with landmarks
#         plt.subplot(1, 2, 1)
#         plt.imshow(image)
#         plt.title("Original Image with Landmarks")
#
#         # Show aligned image
#         plt.subplot(1, 2, 2)
#         plt.imshow(cropped_face)
#         plt.title("Aligned Image")
#
#         plt.show()
#
#         return cropped_face
#     else:
#         # If no face is detected, return the original image
#         plt.imshow(image)
#         plt.title("No Face Detected")
#         plt.show()
#         return image
#
# # Instantiate MTCNN
# mtcnn = MTCNN(keep_all=True, select_largest=True, post_process=False, device=torch.device('cpu'))
#
# # Path to the image file
# image_path = 'C:/Users/LIU/Desktop/face-iv/lhj3.jpg'
# aligned_image = align_and_show_face(image_path, mtcnn)
