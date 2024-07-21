import os
import shutil

'''Merge the rotated images folder into the main folder '''
main_directory = r"D:\yatharth files\SEM 3-2 2024\isl_sop_codes\final_sop_19thfeb\youtube_data"

# Loop through each character from A to Z
for char in range(ord('A'), ord('Z') + 1):
    # Convert character code to character
    character = chr(char)

    # Define the character folder path
    character_folder = os.path.join(main_directory, character)

    # Define the path to the rotated_images folder
    rotated_images_folder = os.path.join(character_folder, "rotated_images")

    # Check if the rotated_images folder exists
    if os.path.exists(rotated_images_folder):
        # Move files from rotated_images folder to character folder
        for file_name in os.listdir(rotated_images_folder):
            # Define source and destination paths
            source_path = os.path.join(rotated_images_folder, file_name)
            destination_path = os.path.join(character_folder, file_name)

            # Move file to character folder
            shutil.move(source_path, destination_path)

        # Remove the rotated_images folder
        os.rmdir(rotated_images_folder)
