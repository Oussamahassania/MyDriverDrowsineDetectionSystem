import cv2
import dlib
import os

# Load the pre-trained shape predictor model and face detector
predictor_path = "shape_predictor_68_face_landmarks.dat"  # Ensure this path is correct
face_detector = dlib.cnn_face_detection_model_v1("mmod_human_face_detector.dat")  # Path to your face detector model
predictor = dlib.shape_predictor(predictor_path)

# Function to detect specific nose landmarks in the image
def detect_and_save_landmarks(image_path, output_dir):
    # Read the image and resize for faster processing
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load image at {image_path}")
        return False

    # Resize image (reduce size to speed up processing)
    scale_factor = 0.5  # Adjust this factor as needed
    image_resized = cv2.resize(image, (0, 0), fx=scale_factor, fy=scale_factor)
    gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    detections = face_detector(gray)

    # Check if any faces are detected
    if not detections:
        print(f"No faces detected in {image_path}")
        return False

    for detection in detections:
        # Get the bounding box from the detection
        face = detection.rect
        landmarks = predictor(gray, face)

        # Extract specific landmarks: 28, 30, 31, 35
        specific_landmarks = [28, 30, 31, 35]
        detected = True

        # Check if all specific landmarks are detected
        for idx in specific_landmarks:
            if landmarks.part(idx) is None:
                detected = False
                break

        if detected:
            # Create a copy of the original image for display
            display_image = image.copy()

            # Draw circles on the detected landmarks in the copy
            for idx in specific_landmarks:
                x = int(landmarks.part(idx).x / scale_factor)  # Scale back coordinates
                y = int(landmarks.part(idx).y / scale_factor)
                cv2.circle(display_image, (x, y), 5, (0, 255, 0), -1)  # Draw a circle on the landmark

            # Display the detected image with landmarks
            cv2.imshow("Detected Image", display_image)
            cv2.waitKey(1)  # Wait for 1 millisecond

            # Save the original image without drawings
            base_filename = os.path.basename(image_path)
            output_path = os.path.join(output_dir, base_filename)
            cv2.imwrite(output_path, image)  # Save the original image
            print(f"Saved detected image to {output_path}")
            return True

    return False

# Directory containing input images
input_dir = r"C:\Users\9o9ie\Desktop\NEW DATA\augmented2_awake"  # Update this path
output_dir = r"C:\Users\9o9ie\Desktop\NEW DATA\detected_awake"  # Update this path

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Process each image in the input directory
for filename in os.listdir(input_dir):
    if filename.endswith(('.png', '.jpg', '.jpeg')):
        image_path = os.path.join(input_dir, filename)
        detect_and_save_landmarks(image_path, output_dir)

# Close all OpenCV windows after processing
cv2.destroyAllWindows()