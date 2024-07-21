import os
from PIL import Image


# To convert all the png screenshots to a jpeg image
def convert_to_jpeg(folder_path):
    # Iterate through each character folder (A to Z)
    for char_folder in os.listdir(folder_path):
        char_folder_path = os.path.join(folder_path, char_folder)

        # Check if the path is a directory
        if os.path.isdir(char_folder_path):
            # Iterate through each image file in the character folder
            for filename in os.listdir(char_folder_path):
                if not filename.lower().endswith(('.jpg', '.jpeg')):
                    # Open the image
                    image_path = os.path.join(char_folder_path, filename)
                    img = Image.open(image_path)

                    # Convert and save the image as JPEG
                    new_filename = os.path.splitext(filename)[0] + '.jpg'
                    new_image_path = os.path.join(char_folder_path, new_filename)
                    img.convert('RGB').save(new_image_path, 'JPEG')

                    # Remove the original non-JPEG image
                    os.remove(image_path)

                    print(f"{filename} converted to JPEG.")


# Specify the folder path
folder_path = r"D:\yatharth files\SEM 3-2 2024\isl_sop_codes\final_sop_19thfeb\youtube_data"

# Call the function to convert non-JPEG images to JPEG format and remove the original ones
convert_to_jpeg(folder_path)
