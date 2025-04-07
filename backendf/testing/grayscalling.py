import cv2
import os

def convert_images_to_grayscale(input_dir, output_dir):
    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Iterate over all files in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith(('.png', '.jpg', '.jpeg')):  # Check for image files
            # Load the image
            image_path = os.path.join(input_dir, filename)
            image = cv2.imread(image_path)

            # Convert the image to grayscale
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Save the grayscale image to the output directory
            gray_image_path = os.path.join(output_dir, filename)
            cv2.imwrite(gray_image_path, gray_image)

            print(f"Converted {filename} to grayscale and saved to {output_dir}")

# Specify the input and output directories
input_directory = r"C:\Users\9o9ie\Desktop\NEW DATA\augmented2_drowsy"  # Update this path
output_directory = r"C:\Users\9o9ie\Desktop\NEW DATA\grey_drowsy"  # Update this path

# Convert images
convert_images_to_grayscale(input_directory, output_directory)