import cv2
import dlib
import numpy as np
import joblib  # Import joblib for loading the model

# Load the pre-trained shape predictor model
predictor_path = "shape_predictor_68_face_landmarks.dat"  # Update this path accordingly
predictor = dlib.shape_predictor(predictor_path)

# Load the face detector model
face_detector = dlib.cnn_face_detection_model_v1("mmod_human_face_detector.dat")  # Update this path accordingly


# Function to extract specific nose landmarks
def extract_specific_nose_landmarks(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces using the CNN face detector
    detections = face_detector(gray)

    if detections:
        # Get the first detected face
        face = detections[0].rect
        landmarks = predictor(gray, face)
        specific_landmarks = [28, 30, 31, 35]  # Landmarks to extract
        result = np.array([[landmarks.part(i).x, landmarks.part(i).y] for i in specific_landmarks]).flatten()
        print(f"Landmarks detected for {image_path}: {result}")  # Debug output
        return result
    else:
        print(f"No faces detected in {image_path}.")  # Debug output
    return None


# Function to predict state based on image
def predict_image(image_path, model):
    landmarks = extract_specific_nose_landmarks(image_path)

    if landmarks is not None:
        landmarks = landmarks.reshape(1, -1)
        prediction = model.predict(landmarks)
        if prediction[0] == 1:
            print("The model predicts: Awake")
        else:
            print("The model predicts: Drowsy")



# Load the trained model
model_filename = "knn_awake_drowsy_classifier_knn.joblib"
clf = joblib.load(model_filename)


test_image_path = r"C:\Users\9o9ie\Desktop\mo0.png"


predict_image(test_image_path, clf)