import os
from PIL import Image


# defining a Python user-defined exception
class Error(Exception):
    """Base class for other exceptions"""

    pass


def convert_image(dir_path, image_name, image_type):
    """
    Convert image into WEBP file format
    :param dir_path: input directory path
    :param image_path: input image path
    :param image_type: input image type
    :return: none
    """
    # opening the image
    full_path = os.path.join(dir_path, image_name)
    im = Image.open(full_path)
    # converting the image to RGBA colour
    im = im.convert("RGBA")
    # spliting the image path (to avoid the .jpg or .png being part of the image name)
    image_name = image_name.split(".")[0]

    # saving the images based upon their specific type:
    if image_type == "jpg" or image_type == "png":
        output_path = os.path.join(dir_path, f"{image_name}.webp")
        im.save(output_path, "webp")
    else:
        # raising an error if we didn't get a jpeg or png file type
        raise Error


def convert():
    """
    Convert PNG, JPG files to WEBP format
    :param: none
    :return: none
    """
    dir_path = "./assets"
    files = os.listdir(dir_path)
    images = [file for file in files if file.endswith(("jpg", "png"))]
    [
        (
            convert_image(dir_path, image, image_type="jpg")
            if image.endswith("jpg")
            else convert_image(dir_path, image, image_type="png")
        )
        for image in images
    ]
    print("[IMG_CONVERTER] All done!")


if __name__ == "__main__":
    convert()
