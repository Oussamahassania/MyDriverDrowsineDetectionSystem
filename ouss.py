import cv2
import numpy as np

# Load an image
image = cv2.imread('C:/Users/dell/Downloads/WhatsApp Image 2025-03-14 at 17.00.14.jpeg')

# Step 1: Resize the image (if necessary)
resized_image = cv2.resize(image, (400, 400))
cv2.imshow('Resized Image', resized_image)

# Step 2: Convert to grayscale
gray_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
cv2.imshow('Grayscale Image', gray_image)

# Step 3: Histogram equalization (if needed to improve lighting)
equalized_image = cv2.equalizeHist(gray_image)
cv2.imshow('Equalized Image', equalized_image)

# Step 4: Face detection (assuming you are using pre-trained HOG or any other detector)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
faces = face_cascade.detectMultiScale(equalized_image, scaleFactor=1.1, minNeighbors=5)

# Draw rectangle around the faces
for (x, y, w, h) in faces:
    cv2.rectangle(resized_image, (x, y), (x+w, y+h), (0, 255, 0), 2)

cv2.imshow('Detected Faces', resized_image)

# Wait until a key is pressed
cv2.waitKey(0)
cv2.destroyAllWindows()
