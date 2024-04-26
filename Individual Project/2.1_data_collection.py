import cv2 as cv
import os
import numpy as np
import json


def detect_faces_with_gpu(save_dir):
    # Load the CNN model
    net = cv.dnn.readNetFromCaffe('deploy.prototxt', 'res10_300x300_ssd_iter_140000.caffemodel')

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Initialize video capture
    cap = cv.VideoCapture(0)

    # Counter for person_id and saved images
    # Dictionary to hold the mapping of labels to people's names
    label_name_map = {}

    person_name = input("Enter person's name: ")
    person_id = input("Enter label number: ")

    # Read existing label-name map from file or create a new one
    map_file_path = os.path.join('C:/Users/LIU/Desktop/face_recog_material/map', "label_name_map.json")
    if os.path.exists(map_file_path):
        with open(map_file_path, "r") as f:
            label_name_map = json.load(f)
    else:
        label_name_map = {}

    label_name_map[int(person_id)] = person_name
    save_counter = 1

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to grayscale
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

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

                # Draw the bounding box of the face along with the associated probability
                text = "{:.2f}%".format(confidence * 100)
                y = startY - 10 if startY - 10 > 10 else startY + 10
                cv.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                cv.putText(frame, text, (startX, y), cv.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)

                key1 = cv.waitKey(1) & 0xFF
                if key1 == ord('p'):  # 'p' has been pressed
                    face_img = gray[startY:endY, startX:endX]
                    person_dir = os.path.join(save_dir, f"s{person_id}")
                    if not os.path.exists(person_dir):
                        os.makedirs(person_dir)  # Create the directory if it does not exist
                    # Construct the save path using the person directory and image count
                    save_path = os.path.join(person_dir, f"{save_counter}.jpg")
                    cv.imwrite(save_path, face_img)
                    print(f"Saved face to {save_path}")
                    save_counter += 1

        # Display the resulting frame
        cv.imshow('Face Detection', frame)

        # Break the loop on 'q' key press
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    with open(os.path.join('C:/Users/LIU/Desktop/face_recog_material/map', "label_name_map.json"), "w") as f:
        json.dump(label_name_map, f, indent=4)

    # When everything is done, release the capture
    cap.release()
    cv.destroyAllWindows()


detect_faces_with_gpu('C:/Users/LIU/Desktop/face_recog_material/datasets')