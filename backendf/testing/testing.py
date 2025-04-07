from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
import imutils
import time
import dlib
import cv2
import numpy as np
import joblib

# Load the pre-trained shape predictor model
predictor_path = "shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(predictor_path)

# Load your trained Logistic Regression model
model_filename = "logistic_regression_awake_drowsy_classifier.joblib"
clf = joblib.load(model_filename)

# Initialize dlib's face detector
detector = dlib.get_frontal_face_detector()

# Initialize the video stream and sleep for a bit
print("[INFO] Initializing camera...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

# Define constants
EYE_AR_THRESH = 0.25
MOUTH_AR_THRESH = 0.79
EYE_AR_CONSEC_FRAMES = 3
COUNTER = 0

# Grab the indexes of the facial landmarks for the eyes and mouth
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mStart, mEnd) = (49, 68)  # Mouth points

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def mouth_aspect_ratio(mouth):
    A = dist.euclidean(mouth[2], mouth[10])
    B = dist.euclidean(mouth[4], mouth[8])
    C = dist.euclidean(mouth[0], mouth[6])
    mar = (A + B) / (2.0 * C)
    return mar

# Video processing loop
frame_count = 0
while True:
    frame = vs.read()
    if frame is None:
        print("[ERROR] Failed to capture frame from the camera.")
        break

    # Increment frame count
    frame_count += 1

    # Process every 2nd frame to improve efficiency
    if frame_count % 2 != 0:
        continue

    # Resize the frame for the model
    model_frame = cv2.resize(frame, (384, 288))
    gray = cv2.cvtColor(model_frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    rects = detector(gray, 0)

    # Display frame at its original size
    display_frame = imutils.resize(frame, width=800)

    if len(rects) > 0:
        for rect in rects:
            # Get facial landmarks
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            # Calculate the center of the face using landmarks
            x_center = (shape[:, 0].min() + shape[:, 0].max()) // 2
            y_center = (shape[:, 1].min() + shape[:, 1].max()) // 2

            # Calculate the width and height of the bounding box
            w = int(shape[:, 0].max() - shape[:, 0].min()) + 20  # Add some padding
            h = int(shape[:, 1].max() - shape[:, 1].min()) + 20  # Add some padding

            # Draw the centered bounding box
            cv2.rectangle(display_frame,
                          (x_center - w // 2, y_center - h // 2),
                          (x_center + w // 2, y_center + h // 2), (0, 255, 0), 2)

            # Calculate EAR
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0

            # Check drowsiness
            if ear < EYE_AR_THRESH:
                COUNTER += 1
                if COUNTER >= EYE_AR_CONSEC_FRAMES:
                    cv2.putText(display_frame, "Drowsy!", (500, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                COUNTER = 0

            # Calculate MAR
            mouth = shape[mStart:mEnd]
            mar = mouth_aspect_ratio(mouth)

            if mar > MOUTH_AR_THRESH:
                cv2.putText(display_frame, "Yawning!", (800, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Prepare features for the model
            features = np.array([ear, mar, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).reshape(1, -1)

            # Make prediction
            try:
                prediction = clf.predict(features)
                if prediction[0] == 1:
                    cv2.putText(display_frame, "Awake", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    cv2.putText(display_frame, "Drowsy", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            except Exception as e:
                print(f"[ERROR] Prediction failed: {e}")
    else:
        # If no face is detected, predict Drowsy
        cv2.putText(display_frame, "Drowsy", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Show the frame at original size
    cv2.imshow("Frame", display_frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

# Cleanup
cv2.destroyAllWindows()
vs.stop()