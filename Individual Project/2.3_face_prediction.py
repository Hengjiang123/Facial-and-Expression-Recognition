import cv2 as cv
import os
import numpy as np
import json


def real_time_face_recognition(model_path, unknown_face_threshold):
    # Load the CNN model for face detection
    net = cv.dnn.readNetFromCaffe('deploy.prototxt', 'res10_300x300_ssd_iter_140000.caffemodel')

    # Load the LBPH face recognizer
    face_recognizer = cv.face.LBPHFaceRecognizer_create()
    # face_recognizer = cv.face.EigenFaceRecognizer_create()
    # face_recognizer = cv.face.FisherFaceRecognizer_create()
    face_recognizer.read(model_path)

    # Initialize video capture
    cap = cv.VideoCapture(0)

    # load map for people and name
    with open(os.path.join('C:/Users/LIU/Desktop/face_recog_material/map', "label_name_map.json"), "r") as f:
        label_name_map = json.load(f)

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        if not ret:
            break

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

            # Filter out weak detections
            if confidence > 0.5:
                # Compute the coordinates of the bounding box
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                grayFaceROI = None
                distance_face = 100
                # Extract the face ROI
                if (endX - startX) > 0 and (endY - startY) > 0:  # Ensure the ROI is not empty
                    faceROI = frame[startY:endY, startX:endX]
                    if faceROI.size != 0:  # Additional check for emptyness
                        grayFaceROI = cv.cvtColor(faceROI, cv.COLOR_BGR2GRAY)
                        # resize the image
                        desired_size = (200, 200)
                        grayFaceROI = cv.resize(grayFaceROI, desired_size)

                # Predict the face
                # distance_face indicates the distance between the test image
                # and the nearest neighbouring image in the training set,
                # with lower values indicating a higher match
                if grayFaceROI is not None and grayFaceROI.size > 0 and grayFaceROI.shape[0] > 0 and grayFaceROI.shape[1] > 0:
                    try:
                        label, distance_face = face_recognizer.predict(grayFaceROI)
                        person_name = label_name_map.get(str(label), "Unknown")
                        # Display the label and bounding box
                        if distance_face > unknown_face_threshold:
                            label_text = f"Unknown    Distance: {distance_face:.2f}"
                        else:
                            label_text = f"Name: {label}  Distance: {distance_face:.2f}"
                        cv.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                        cv.putText(frame, label_text, (startX, startY - 10), cv.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255),
                                   2)
                    except cv.error as e:
                        print(f"An error occurred during prediction: {e}")

        # Display the resulting frame
        cv.imshow('Real-time Face Recognition', frame)

        # Break the loop on 'q' key press
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture
    cap.release()
    cv.destroyAllWindows()


model_path = 'C:/Users/LIU/Desktop/face_recog_material/model/trained_model_LBPH_orl.xml'       # 63 threshold
# model_path = 'C:/Users/LIU/Desktop/face_recog_material/model/trained_model_Eigen.xml'      # 7000
# model_path = 'C:/Users/LIU/Desktop/face_recog_material/model/trained_model_Fisher.xml'     # 800

# Call the function with the model path
real_time_face_recognition(model_path, 63)