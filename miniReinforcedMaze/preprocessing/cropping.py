from PIL import Image
import numpy as np

def crop(input_path, image_names, height=16, width=16, sprite_size=32, show=False, return_indexes=None):
    """
    Crop the images from the given path and names into multiple pieces of the given size
    and return a list of cropped and resized images.
    If show is True, display the resized images.
    If return_indexes is provided, return a tuple of the cropped and resized images along with the specified indexes.
    """
    images = []  # List to store the cropped and resized images
    indexes = []  # List to store the indexes

    for image_name in image_names:
        image_path = (
            f"{input_path}/{image_name}"  # Construct the full path of the image
        )
        im = Image.open(image_path)  # Open the image
        imgwidth, imgheight = im.size  # Get the dimensions of the image

        # Iterate through the image in steps of given height and width
        for i in range(0, imgheight, height):
            for j in range(0, imgwidth, width):
                box = (j, i, j + width, i + height)  # Define the cropping box
                cropped_image = im.crop(box)  # Crop the image using the box coordinates
                resized_image = cropped_image.resize(
                    (sprite_size, sprite_size),
                    Image.ANTIALIAS)  # Resize the cropped image

                if show:
                    resized_image.show()  # Display the resized image if show is True

                images.append(resized_image)  # Add the resized image to the list

                if return_indexes and len(images) - 1 in return_indexes:
                    indexes.append(
                        len(images) - 1
                    )  # Append the index of the added image

    if return_indexes is not None:
        if -1 in return_indexes:
            indexes.append(-1)  # Include index -1 in the indexes list
        return_images = [images[i] for i in indexes]
        return return_images, indexes
    else:
        return images  # Return the list of cropped and resized images
