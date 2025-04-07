import os
import shutil

# Define the source and destination directories
source_dir = r'C:\Users\9o9ie\Desktop\detected\augmented_awake'
destination_dir = r'C:\Users\9o9ie\Desktop\detected\awake'

# Ensure the destination directory exists
os.makedirs(destination_dir, exist_ok=True)

# Get a list of all image files in the source directory
files = [file for file in os.listdir(source_dir) if file.endswith(('.png', '.jpg', '.jpeg', '.gif'))]

# Limit to the first 4000 files
files_to_move = files[:4000]

# Loop through the files and move them
for file_name in files_to_move:
    source_file = os.path.join(source_dir, file_name)
    destination_file = os.path.join(destination_dir, file_name)

    # Move the file
    shutil.move(source_file, destination_file)

print(f"Moved {len(files_to_move)} images successfully.")