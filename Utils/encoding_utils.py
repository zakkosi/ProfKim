# encoding_utils.py

import json
from PIL import Image
import torch
import clip
from torchvision import transforms
from dotenv import load_dotenv
load_dotenv()
import openai
import os

openai.api_key = os.getenv("OPENAI_API_KEY")

# CLIP 모델 로딩 (이미지 임베딩용)
clip_model, clip_preprocess = clip.load("ViT-B/32")
clip_model.eval()

# OpenAI text embedding (ada-002)
def encode_text_ada002(text: str):
    try:
        response = openai.Embedding.create(
            model="text-embedding-ada-002",
            input=text
        )
        return torch.tensor(response["data"][0]["embedding"])
    except Exception as e:
        raise RuntimeError(f"[텍스트 임베딩 실패] {e}")

# CLIP 이미지 임베딩
def encode_image_clip(image_tensor):
    device = clip_model.visual.proj.device 
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        return clip_model.encode_image(image_tensor.unsqueeze(0)).squeeze(0)

# 이미지 로딩
from Utils.image_utils import load_image_as_tensor

# 슬라이딩 윈도우 생성 함수
def generate_sliding_windows(pdf_dir: str, text_json_path: str, window_size: int = 3):
    with open(text_json_path, 'r', encoding='utf-8') as f:
        page_texts = json.load(f)  # {"1": "내용", "2": "내용" ...}

    filenames = sorted([fn for fn in os.listdir(pdf_dir) if fn.endswith('.jpg')])
    filenames = sorted(filenames, key=lambda x: int(x.split('_')[1].split('.')[0]))

    windows = []
    for i in range(len(filenames) - window_size + 1):
        window_files = filenames[i:i + window_size]
        page_nums = [int(f.split('_')[1].split('.')[0]) for f in window_files]
        texts = [page_texts.get(str(pn), "") for pn in page_nums]
        combined_text = "\n".join(texts)

        center_img = window_files[window_size // 2]  # 중앙 이미지

        windows.append({
            "doc_id": f"pages_{str(page_nums[0]).zfill(3)}_{str(page_nums[-1]).zfill(3)}",
            "image_path": center_img,
            "text": combined_text,
            "pages": page_nums
        })
    return windows
