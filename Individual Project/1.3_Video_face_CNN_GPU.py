# 1. 实时检测人脸

# import cv2 as cv
# import numpy as np
#
#
# def detect_faces_with_gpu():
#     # Load the CNN model
#     net = cv.dnn.readNetFromCaffe('deploy.prototxt', 'res10_300x300_ssd_iter_140000.caffemodel')
#
#     # # Use CUDA as the preferable backend and target
#     # net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
#     # net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)
#
#     # Initialize video capture
#     cap = cv.VideoCapture(0)
#
#     while True:
#         # Capture frame-by-frame
#         ret, frame = cap.read()
#         if not ret:
#             break
#
#         # Get the frame dimensions
#         (h, w) = frame.shape[:2]
#
#         # Preprocess the frame for the neural network
#         blob = cv.dnn.blobFromImage(cv.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
#
#         # Pass the blob through the network and obtain the detections
#         net.setInput(blob)
#         detections = net.forward()
#
#         # Loop over the detections
#         for i in range(0, detections.shape[2]):
#             # Extract the confidence of the detection
#             confidence = detections[0, 0, i, 2]
#
#             # Filter out weak detections by ensuring the confidence is greater than a minimum threshold
#             if confidence > 0.5:
#                 # Compute the coordinates of the bounding box for the face
#                 box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
#                 (startX, startY, endX, endY) = box.astype("int")
#
#                 # Draw the bounding box of the face along with the associated probability
#                 text = "{:.2f}%".format(confidence * 100)
#                 y = startY - 10 if startY - 10 > 10 else startY + 10
#                 cv.rectangle(frame, (startX, startY), (endX, endY), (255, 0, 0), 2)
#                 cv.putText(frame, text, (startX, y), cv.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
#
#         # Display the resulting frame
#         cv.imshow('Face Detection', frame)
#
#         # Break the loop on 'q' key press
#         if cv.waitKey(1) & 0xFF == ord('q'):
#             break
#
#     # When everything is done, release the capture
#     cap.release()
#     cv.destroyAllWindows()
#
#
# detect_faces_with_gpu()





#
# #2.  检测人脸并绘制置信度和帧率
#
# import cv2 as cv
# import numpy as np
# import time
# import matplotlib.pyplot as plt
#
# def detect_faces_and_plot():
#     # Load the CNN model
#     net = cv.dnn.readNetFromCaffe('deploy.prototxt', 'res10_300x300_ssd_iter_140000.caffemodel')
#
#     # Uncomment these lines if you are using a GPU with CUDA support
#     # net.setPreferableBackend(cv.dnn.DNN_BACKEND_CUDA)
#     # net.setPreferableTarget(cv.dnn.DNN_TARGET_CUDA)
#
#     # Initialize video capture
#     cap = cv.VideoCapture(0)
#
#     # Initialize timing and recording
#     start_time = time.time()
#     frame_rates = []
#     confidences = []
#
#     while True:
#         # Check elapsed time
#         elapsed_time = time.time() - start_time
#         if elapsed_time > 60:  # Stop after 60 seconds
#             break
#
#         # Capture frame-by-frame
#         ret, frame = cap.read()
#         if not ret:
#             break
#
#         # Frame timing
#         frame_start_time = time.time()
#
#         # Get the frame dimensions
#         (h, w) = frame.shape[:2]
#
#         # Preprocess the frame for the neural network
#         blob = cv.dnn.blobFromImage(cv.resize(frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
#
#         # Pass the blob through the network and obtain the detections
#         net.setInput(blob)
#         detections = net.forward()
#
#         # Track the highest confidence
#         max_confidence = 0
#
#         # Loop over the detections
#         for i in range(0, detections.shape[2]):
#             # Extract the confidence of the detection
#             confidence = detections[0, 0, i, 2]
#             max_confidence = max(max_confidence, confidence)
#
#             if confidence > 0.5:
#                 # Compute the coordinates of the bounding box for the face
#                 box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
#                 (startX, startY, endX, endY) = box.astype("int")
#
#                 # Draw the bounding box of the face along with the associated probability
#                 text = "{:.2f}%".format(confidence * 100)
#                 y = startY - 10 if startY - 10 > 10 else startY + 10
#                 cv.rectangle(frame, (startX, startY), (endX, endY), (255, 0, 0), 2)
#                 cv.putText(frame, text, (startX, y), cv.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
#
#         # Calculate and record frame rate
#         frame_end_time = time.time()
#         frame_rate = 1.0 / (frame_end_time - frame_start_time)
#         frame_rates.append(frame_rate)
#         confidences.append(max_confidence)
#
#         # Display the resulting frame
#         cv.imshow('Face Detection', frame)
#
#         # Break the loop on 'q' key press
#         if cv.waitKey(1) & 0xFF == ord('q'):
#             break
#
#     # When everything is done, release the capture
#     cap.release()
#     cv.destroyAllWindows()
#
#     # Plotting
#     plt.figure()
#     ax1 = plt.gca()
#     ax2 = ax1.twinx()
#     ax1.plot(frame_rates, 'b-')
#     ax2.plot([c * 100 for c in confidences], 'r-')
#
#     ax1.set_xlabel('Time (s)')
#     ax1.set_ylabel('Frame Rate (fps)', color='b')
#     ax2.set_ylabel('Confidence (%)', color='r')
#
#     ax1.set_ylim(0, 60)  # Setting Y axis limits for frame rate
#     ax2.set_ylim(80, 110)  # Setting Y axis limits for confidence
#
#     plt.title('Frame Rate and Confidence Over Time')
#     plt.show()
#
# detect_faces_and_plot()






#
# # 3 检测一张图片中的人脸：
#
# import cv2 as cv
# import numpy as np
# import matplotlib.pyplot as plt
#
# def detect_faces_in_image(image_path):
#     # Load the CNN model
#     net = cv.dnn.readNetFromCaffe('deploy.prototxt', 'res10_300x300_ssd_iter_140000.caffemodel')
#
#     # Load image
#     image = cv.imread(image_path)
#     if image is None:
#         print("Could not read image.")
#         return
#
#     # Get the frame dimensions
#     (h, w) = image.shape[:2]
#
#     # Preprocess the frame for the neural network
#     blob = cv.dnn.blobFromImage(cv.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
#
#     # Pass the blob through the network and obtain the detections
#     net.setInput(blob)
#     detections = net.forward()
#
#     # Loop over the detections
#     for i in range(0, detections.shape[2]):
#         # Extract the confidence of the detection
#         confidence = detections[0, 0, i, 2]
#
#         if confidence > 0.5:  # Filter out weak detections by ensuring the confidence is greater than 50%
#             # Compute the coordinates of the bounding box for the face
#             box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
#             (startX, startY, endX, endY) = box.astype("int")
#
#             # Draw the bounding box of the face along with the associated probability
#             text = "{:.2f}%".format(confidence * 100)
#             y = startY - 10 if startY - 10 > 10 else startY + 10
#             cv.rectangle(image, (startX, startY), (endX, endY), (255, 0, 0), 2)
#             cv.putText(image, text, (startX, y), cv.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2)
#
#     # Display the resulting image
#     cv.imshow('Face Detection', image)
#     cv.waitKey(0)
#     cv.destroyAllWindows()
#
# # Specify the path to your image
# image_path = 'C:/Users/LIU/Desktop/archive2/train/happy/Training_4791031.jpg'
# detect_faces_in_image(image_path)








#
# # 4 检测并删除文件的子文件中不是人脸的图片
#
# import os
# import cv2 as cv
# import numpy as np
#
# def detect_faces_in_image(image_path, net):
#     # Load image
#     image = cv.imread(image_path)
#     if image is None:
#         print("Could not read image.")
#         return False
#
#     # Get the frame dimensions
#     (h, w) = image.shape[:2]
#
#     # Preprocess the frame for the neural network
#     blob = cv.dnn.blobFromImage(cv.resize(image, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0))
#
#     # Pass the blob through the network and obtain the detections
#     net.setInput(blob)
#     detections = net.forward()
#
#     # Loop over the detections and check if there's any face
#     for i in range(0, detections.shape[2]):
#         confidence = detections[0, 0, i, 2]
#         if confidence > 0.99:
#             return True  # Face detected
#
#     return False  # No face detected
#
# def delete_images_without_faces(directory_path):
#     # Load the CNN model
#     net = cv.dnn.readNetFromCaffe('deploy.prototxt', 'res10_300x300_ssd_iter_140000.caffemodel')
#     count_deleted = 0
#
#     # Walk through all files in the directory that contains the subdirectories
#     for subdir, dirs, files in os.walk(directory_path):
#         for file in files:
#             if file.lower().endswith(('.png', '.jpg', '.jpeg')):
#                 file_path = os.path.join(subdir, file)
#                 if not detect_faces_in_image(file_path, net):
#                     os.remove(file_path)
#                     count_deleted += 1
#                     print(f"Deleted {file_path}")
#
#     return count_deleted
#
# # Specify the path to your directory
# directory_path = 'C:/Users/LIU/Desktop/CASIA_delet_smaller_than_50_21 - Copy/train'
# deleted_count = delete_images_without_faces(directory_path)
# print(f"Total deleted images: {deleted_count}")







# 统计文件中图片数

import os

def analyze_images(directory):
    # 初始化变量以存储统计结果
    min_images = float('inf')
    max_images = 0
    total_images = 0
    num_folders = 0

    # 遍历目录中的所有子文件夹
    for folder in os.listdir(directory):
        folder_path = os.path.join(directory, folder)
        if os.path.isdir(folder_path):
            num_folders += 1
            # 获取当前文件夹中的图片数量
            images_count = len([name for name in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, name))])
            # 更新最大和最小图片数量记录
            if images_count > max_images:
                max_images = images_count
            if images_count < min_images:
                min_images = images_count
            # 累加总图片数
            total_images += images_count

    # 如果没有找到任何子文件夹，调整最小图片数量为0
    if num_folders == 0:
        min_images = 0

    return min_images, max_images, total_images

# 指定你的文件夹路径
train_dir = 'C:/Users/LIU/Desktop/CASIA_delet_smaller_than_0.99/train'

# 调用函数并打印结果
min_images, max_images, total_images = analyze_images(train_dir)
print(f"最少图片数: {min_images}")
print(f"最多图片数: {max_images}")
print(f"总图片数: {total_images}")
