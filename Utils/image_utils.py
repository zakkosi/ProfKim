# utils/image_utils.py
from PIL import Image
import io

def load_image_as_bytes(image_path):
    with Image.open(image_path).convert("RGB") as img:
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG')
        return img_byte_arr.getvalue()

def crop_image(image, bbox):
    # bbox: (left, upper, right, lower)
    return image.crop(bbox)


def resize_image(image, size=(224, 224)):
    return image.resize(size)
