import os
import shutil

# Define the paths
master_folder = r'D:\yatharth files\SEM 3-2 2024\isl_sop_codes\final_sop_19thfeb\combined_dataset_new'
new_folders = [r'D:\yatharth files\SEM 3-2 2024\isl_sop_codes\final_sop_19thfeb\4waydatasetnew\split1', r'D:\yatharth '
                                                                                                     r'files\SEM 3-2 '
                                                                                                     r'2024'
                                                                                                     r'\isl_sop_codes\final_sop_19thfeb\4waydatasetnew\split2', r'D:\yatharth files\SEM 3-2 2024\isl_sop_codes\final_sop_19thfeb\4waydatasetnew\split3', r'D:\yatharth files\SEM 3-2 2024\isl_sop_codes\final_sop_19thfeb\4waydatasetnew\split4']
for new_folder in new_folders:
    os.makedirs(new_folder, exist_ok=True)
    for char in range(65, 91):  # ASCII range for A-Z
        os.makedirs(os.path.join(new_folder, chr(char)), exist_ok=True)

# Initialize a counter for round-robin distribution
counter = 0

# Iterate over each character folder in the master folder
for char_folder in os.listdir(master_folder):
    char_folder_path = os.path.join(master_folder, char_folder)
    if os.path.isdir(char_folder_path):
        # Get list of image files in the character folder
        image_files = [f for f in os.listdir(char_folder_path) if os.path.isfile(os.path.join(char_folder_path, f))]

        # Distribute image files to corresponding subfolders in new folders (round-robin)
        for image_file in image_files:
            destination_folder = new_folders[counter % len(new_folders)]  # Round-robin distribution
            shutil.copy(os.path.join(char_folder_path, image_file), os.path.join(destination_folder, char_folder))
            counter += 1  # Increment counter for next image file