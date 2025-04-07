import os
import numpy as np
import cv2  # Ensure OpenCV is installed
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split

# Adjust these to your expected feature dimensions
EXPECTED_FEATURE_COUNT = 8


def extract_specific_nose_landmarks(image_path):
    # Load the image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Image at {image_path} could not be loaded.")

    # Preprocess the image (resize to a fixed size)
    img = cv2.resize(img, (100, 100))  # Example size, adjust as needed

    # Example feature extraction logic (replace with your actual logic)
    features = np.array([
        np.mean(img),  # Mean pixel value
        np.var(img),  # Variance
        np.max(img),  # Max pixel value
        np.min(img),  # Min pixel value
        np.sum(img),  # Sum of pixel values
        np.std(img),  # Standard deviation
        np.median(img),  # Median pixel value
        np.count_nonzero(img)  # Count of non-zero pixels
    ])

    if features.shape[0] != EXPECTED_FEATURE_COUNT:
        raise ValueError(f"Expected {EXPECTED_FEATURE_COUNT} features, got {features.shape[0]}.")

    return features


def load_dataset(awake_dir, drowsy_dir):
    features = []
    labels = []

    # Load awake images
    for filename in os.listdir(awake_dir):
        img_path = os.path.join(awake_dir, filename)
        if os.path.isfile(img_path):
            feature = extract_specific_nose_landmarks(img_path)
            features.append(feature)
            labels.append(1)  # 1 for awake

    # Load drowsy images
    for filename in os.listdir(drowsy_dir):
        img_path = os.path.join(drowsy_dir, filename)
        if os.path.isfile(img_path):
            feature = extract_specific_nose_landmarks(img_path)
            features.append(feature)
            labels.append(0)  # 0 for drowsy

    # Convert to numpy arrays
    features = np.array(features)
    labels = np.array(labels)

    print(f"Loaded {len(features)} samples.")  # Debug statement
    return features, labels


# Load the trained model
model_filename = "awake_drowsy_classifier.joblib"  # Update this path as needed
clf = joblib.load(model_filename)

# Specify your dataset directories
awake_dir = r"C:\Users\9o9ie\Desktop\NEW DATA\augmented2_awake"  # Update this path
drowsy_dir = r"C:\Users\9o9ie\Desktop\NEW DATA\augmented2_drowsy"  # Update this path

# Load the dataset
X, y = load_dataset(awake_dir, drowsy_dir)

# Check if dataset is empty
if X.size == 0 or y.size == 0:
    raise ValueError("No samples were loaded. Please check your dataset paths.")

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Make predictions on the test set
y_pred = clf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

# Print detailed classification report
print("Classification report:")
print(classification_report(y_test, y_pred))

# Compute the confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)

# Plotting the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Drowsy', 'Awake'],
            yticklabels=['Drowsy', 'Awake'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title('Confusion Matrix')
plt.show()