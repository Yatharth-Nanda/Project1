from PIL import Image
import os
import shutil

'''Program used to rotate images by given angles and store them in a folder '''


def rotate_images(folder_path):
    # Define rotation degrees
    degrees = [ 7,10  ]

    # Create a directory to save rotated images
    save_folder = os.path.join(folder_path, "rotated_images")

    # Delete pre-existing "rotated_images" folder
    if os.path.exists(save_folder):
        shutil.rmtree(save_folder)

    # Create new "rotated_images" folder
    os.makedirs(save_folder)

    # Iterate through each file in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith(('.jpg', '.jpeg', '.png', '.gif')):  # Adjust extensions as needed
            file_path = os.path.join(folder_path, filename)

            # Open the image
            img = Image.open(file_path)

            # Iterate through each degree
            for degree in degrees:
                # Rotate clockwise
                rotated_img_clockwise = img.rotate(degree)

                # Save rotated image with clockwise direction
                save_path_clockwise = os.path.join(save_folder, f"{filename[:-4]}_clock_{degree}.jpg")
                rotated_img_clockwise.save(save_path_clockwise)

                # Rotate anticlockwise
                rotated_img_anticlockwise = img.rotate(-degree)

                # Save rotated image with anticlockwise direction
                save_path_anticlockwise = os.path.join(save_folder, f"{filename[:-4]}_anticlock_{degree}.jpg")
                rotated_img_anticlockwise.save(save_path_anticlockwise)

                print(f"Rotated and saved {filename} clockwise and anticlockwise by {degree} degrees.")


# Example usage


main_folder_path = r"D:\yatharth files\SEM 3-2 2024\isl_sop_codes\final_sop_19thfeb\youtube_data"
for letter in range(ord('A'), ord('Z') + 1):
    folder_path = f"{main_folder_path}\\{chr(letter)}"
    rotate_images(folder_path)
