import os
from PIL import Image, UnidentifiedImageError
from iam_ondb._utils import file_iterator, get_logger


def get_image_data(path):
    im = Image.open(path)

    res = im.copy()
    im.close()
    return res


def reshape(im, width, height):
    pixel_list = im.getdata()
    image_data = []
    for _ in range(height):
        image_data.append([0] * width)

    for i, pixel_value in enumerate(pixel_list):
        y = get_row(i, width)
        x = get_column(i, width)

        image_data[y][x] = pixel_value
    return image_data


def get_row(pixel_index, image_width):
    return pixel_index // image_width


def get_column(pixel_index, image_width):
    return pixel_index % image_width


def images_iterator(images_dir):
    for path in file_iterator(images_dir):
        image_data = try_getting_image_data(path)
        if image_data is not None:
            yield image_data


def try_getting_image_data(path):
    logger = get_logger()

    _, ext = os.path.splitext(path)
    if ext in ['.tif', '.jpg', '.png']:
        try:
            return get_image_data(path)
        except UnidentifiedImageError as e:
            logger.error(repr(e))
        except:
            logger.exception('Unknown error whilst getting an image data')
