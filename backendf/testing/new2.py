import cv2
import dlib
import numpy as np
import joblib
from scipy.spatial import distance as dist
from imutils.video import VideoStream
import time

# Load the pre-trained shape predictor model
predictor_path = "shape_predictor_68_face_landmarks.dat"  # Update this path accordingly
predictor = dlib.shape_predictor(predictor_path)

# Load the trained logistic regression model for head position
model_filename = "logistic_regression_awake_drowsy_classifier.joblib"  # Update this path
clf = joblib.load(model_filename)

# Load the face detector model
face_detector =dlib.cnn_face_detection_model_v1("mmod_human_face_detector.dat")  # Update this path accordingly

# Initialize video stream
vs = VideoStream(src=0).start()
time.sleep(2.0)

# Define thresholds for eye aspect ratio
EYE_AR_THRESH = 0.25
EYE_AR_CONSEC_FRAMES = 3
COUNTER = 0

# Grab the indexes of the facial landmarks for the eyes
(lStart, lEnd) = (36, 42)  # Indices for the left eye
(rStart, rEnd) = (42, 48)  # Indices for the right eye
nose_landmarks_indices = [28, 30, 31, 35]  # Nose landmarks for head position


def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)


def extract_nose_landmarks(dlib_shape):
    return np.array([[dlib_shape.part(i).x, dlib_shape.part(i).y] for i in nose_landmarks_indices]).flatten()


while True:
    frame = vs.read()
    if frame is None:
        print("[ERROR] Failed to capture frame from the camera.")
        break

    # Resize the frame
    resized_frame = cv2.resize(frame, (384, 288))

    # Convert to RGB (required for CNN-based face detector)
    rgb = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)

    # Ensure image is C-contiguous and uint8
    rgb = np.require(rgb, np.uint8, 'C')

    # Debugging print
    print(f"RGB dtype: {rgb.dtype}, shape: {rgb.shape}")

    # Detect faces using CNN model
    detections = face_detector(rgb)

    if detections:
        for detection in detections:
            face = detection.rect
            gray = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale for shape predictor
            shape = predictor(gray, face)

            # Extract nose landmarks for head position detection
            nose_features = extract_nose_landmarks(shape)
            prediction = clf.predict([nose_features])

            label = "Awake" if prediction[0] == 1 else "Drowsy"
            color = (0, 255, 0) if prediction[0] == 1 else (0, 0, 255)
            cv2.putText(resized_frame, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

            # Extract left and right eye landmarks
            leftEye = np.array([[shape.part(i).x, shape.part(i).y] for i in range(lStart, lEnd)])
            rightEye = np.array([[shape.part(i).x, shape.part(i).y] for i in range(rStart, rEnd)])
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0

            if ear < EYE_AR_THRESH:
                COUNTER += 1
                if COUNTER >= EYE_AR_CONSEC_FRAMES:
                    cv2.putText(resized_frame, "Drowsy (Eye Blinking)", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                (0, 0, 255), 2)
            else:
                COUNTER = 0

            # Draw eye contours
            cv2.drawContours(resized_frame, [cv2.convexHull(leftEye)], -1, (0, 255, 0), 1)
            cv2.drawContours(resized_frame, [cv2.convexHull(rightEye)], -1, (0, 255, 0), 1)

    cv2.imshow("Frame", resized_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()
