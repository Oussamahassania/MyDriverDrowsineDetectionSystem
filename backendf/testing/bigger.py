import cv2
import os
import imgaug.augmenters as iaa

# Directory for awake images
awake_dir = r"C:\Users\9o9ie\Desktop\NEW DATA\augmented_drowsy"

# Augmentation sequence with conservative settings
aug = iaa.Sequential([
    iaa.Affine(rotate=(-10, 10)),  # Rotate between -10 and 10 degrees
    iaa.Multiply((0.9, 1.1)),  # Change brightness slightly
    iaa.AddToHueAndSaturation((-10, 10)),  # Change hue and saturation slightly
    iaa.Crop(percent=(0, 0.05)),  # Randomly crop up to 5%
    iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=(0, 1.0))),  # Gaussian blur with lower sigma
    iaa.Sometimes(0.5, iaa.Dropout(p=(0.01, 0.05)))  # Less dropout for noise
])

def augment_and_save(image_path, output_dir, augment_count=10):
    im = cv2.imread(image_path)
    if im is None:
        print(f"Warning: Unable to load image {image_path}")
        return

    # Create a base filename for augmented images
    base_filename = os.path.splitext(os.path.basename(image_path))[0]

    # Perform augmentation
    for i in range(augment_count):
        augmented_image = aug(image=im)
        output_path = os.path.join(output_dir, f"{base_filename}_aug_{i}.jpg")
        cv2.imwrite(output_path, augmented_image)
        print(f"Saved augmented image to {output_path}")

# Create output directory for augmented images
augmented_awake_dir = r"C:\Users\9o9ie\Desktop\NEW DATA\augmented2_drowsy"
os.makedirs(augmented_awake_dir, exist_ok=True)

# Augment awake images
for filename in os.listdir(awake_dir):
    if filename.endswith(('.png', '.jpg', '.jpeg')):
        augment_and_save(os.path.join(awake_dir, filename), augmented_awake_dir)