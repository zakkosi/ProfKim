# utils/image_utils.py (기존 함수 + CLIP 전처리용 함수 추가)

from PIL import Image
import io
import torch
from torchvision import transforms

def load_image_as_bytes(image_path):
    with Image.open(image_path).convert("RGB") as img:
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='PNG')
        return img_byte_arr.getvalue()

def crop_image(image, bbox):
    return image.crop(bbox)

def resize_image(image, size=(224, 224)):
    return image.resize(size)

# ✅ CLIP 전처리용 (추가)
clip_preprocess = transforms.Compose([
    transforms.Resize(224, interpolation=Image.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073],
                         std=[0.26862954, 0.26130258, 0.27577711])
])

def load_image_resized(image_path, size=(224, 224)):
    with Image.open(image_path).convert("RGB") as img:
        return img.resize(size)

def load_image_as_tensor(image_path):
    img = Image.open(image_path).convert("RGB")
    return clip_preprocess(img)
