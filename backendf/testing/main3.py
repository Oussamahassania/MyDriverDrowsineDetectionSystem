import cv2
import dlib
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression  # Import Logistic Regression
from sklearn.metrics import classification_report, accuracy_score
import joblib  # Import joblib for saving the model

# Load the pre-trained shape predictor model
predictor_path = "shape_predictor_68_face_landmarks.dat"  # Update this path accordingly
predictor = dlib.shape_predictor(predictor_path)

# Load the face detector model
face_detector = dlib.cnn_face_detection_model_v1("mmod_human_face_detector.dat")  # Update this path accordingly

# Function to extract specific nose landmarks
def extract_specific_nose_landmarks(image_path):
    # Load the image in grayscale directly
    gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Detect faces using the CNN face detector
    detections = face_detector(gray)

    if detections:
        # Get the first detected face
        face = detections[0].rect
        landmarks = predictor(gray, face)
        specific_landmarks = [28, 30, 31, 35]  # Landmarks to extract
        return np.array([[landmarks.part(i).x, landmarks.part(i).y] for i in specific_landmarks]).flatten()
    return None

# Load your dataset from separate folders
def load_dataset(awake_dir, drowsy_dir):
    features = []
    labels = []

    # Load awake images
    print("Loading awake images...")
    awake_files = [f for f in os.listdir(awake_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    total_awake = len(awake_files)

    for idx, filename in enumerate(awake_files):
        image_path = os.path.join(awake_dir, filename)
        print(f"Loading awake image {idx + 1}/{total_awake}: {filename}")  # Real-time feedback
        landmarks = extract_specific_nose_landmarks(image_path)
        if landmarks is not None:
            features.append(landmarks)
            labels.append(1)  # Label for awake
        else:
            print(f"Warning: No landmarks found for {filename}")

    # Load drowsy images
    print("Loading drowsy images...")
    drowsy_files = [f for f in os.listdir(drowsy_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    total_drowsy = len(drowsy_files)

    for idx, filename in enumerate(drowsy_files):
        image_path = os.path.join(drowsy_dir, filename)
        print(f"Loading drowsy image {idx + 1}/{total_drowsy}: {filename}")  # Real-time feedback
        landmarks = extract_specific_nose_landmarks(image_path)
        if landmarks is not None:
            features.append(landmarks)
            labels.append(0)  # Label for drowsy
        else:
            print(f"Warning: No landmarks found for {filename}")

    print("Dataset loading complete.")
    return np.array(features), np.array(labels)

# Specify the directories for awake and drowsy images
awake_dir = r"C:\Users\9o9ie\Desktop\NEW DATA\grey_awake"  # Update this path
drowsy_dir = r"C:\Users\9o9ie\Desktop\NEW DATA\grey_drowsy"  # Update this path

# Load dataset
X, y = load_dataset(awake_dir, drowsy_dir)

# Split the dataset into training and testing sets
print("Splitting dataset into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Logistic Regression Classifier
print("Training the model...")
clf = LogisticRegression(max_iter=1000)  # Increase max_iter if needed
clf.fit(X_train, y_train)
print("Model training complete.")

# Make predictions on the test set
print("Making predictions on the test set...")
y_pred = clf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Print detailed classification report
print("Classification report:")
print(classification_report(y_test, y_pred))

# Save the trained model to the working directory
model_filename = "logistic_regression_awake_drowsy_classifier.joblib"  # Updated filename for Logistic Regression model
joblib.dump(clf, model_filename)
print(f"Model saved as {model_filename}")