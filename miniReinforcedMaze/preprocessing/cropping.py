from PIL import Image
import os
from typing import List, Tuple, Optional


def crop(
    input_path: str,
    image_names: List[str],
    height: int = 16,
    width: int = 16,
    sprite_size: int = 32,
    show: bool = False,
    return_indexes: Optional[List[int]] = None,
) -> List[Image.Image] | Tuple[List[Image.Image], List[int]]:
    """
    Crop and resize images from the given path and names.

    Args:
        input_path (str): Path to the directory containing the images.
        image_names (List[str]): List of image file names to process.
        height (int): Height of each crop. Defaults to 16.
        width (int): Width of each crop. Defaults to 16.
        sprite_size (int): Size to which cropped images will be resized. Defaults to 32.
        show (bool): If True, display the resized images. Defaults to False.
        return_indexes (Optional[List[int]]): If provided, return only the specified indexes.

    Returns:
        If return_indexes is None:
            List[Image.Image]: List of cropped and resized images.
        If return_indexes is provided:
            Tuple[List[Image.Image], List[int]]: Tuple of (cropped and resized images, indexes).
    """
    images = []
    indexes = []

    for image_name in image_names:
        image_path = os.path.join(input_path, image_name)
        with Image.open(image_path) as im:
            imgwidth, imgheight = im.size
            for i in range(0, imgheight, height):
                for j in range(0, imgwidth, width):
                    box = (j, i, j + width, i + height)
                    cropped_image = im.crop(box)
                    resized_image = cropped_image.resize(
                        (sprite_size, sprite_size), Image.LANCZOS
                    )

                    if show:
                        resized_image.show()

                    images.append(resized_image)

                    if return_indexes and len(images) - 1 in return_indexes:
                        indexes.append(len(images) - 1)

    if return_indexes is not None:
        if -1 in return_indexes:
            indexes.append(-1)
        return [images[i] for i in indexes], indexes
    else:
        return images

