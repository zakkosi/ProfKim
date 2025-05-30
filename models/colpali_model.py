# models/colpali_model.py

import torch
import numpy as np
from PIL import Image
from colpali_engine.models import ColQwen2, ColQwen2Processor
from transformers.utils import is_flash_attn_2_available

# ---------------------------
# 1. 모델 및 전처리기 초기화
# ---------------------------
print("[ColPali] 모델 및 프로세서 로딩 중...")

llm_model = ColQwen2.from_pretrained(
    "vidore/colqwen2-v1.0",
    torch_dtype=torch.bfloat16,
    device_map="cuda:0",
    attn_implementation="flash_attention_2" if is_flash_attn_2_available() else None,
).eval()

processor = ColQwen2Processor.from_pretrained("vidore/colqwen2-v1.0")

# ---------------------------
# 2. 이미지 임베딩 함수
# ---------------------------
def encode_image(image: Image.Image) -> np.ndarray:
    """
    ColQwen2를 사용하여 이미지 임베딩을 생성함
    Args:
        image: PIL.Image (224x224 등)
    Returns:
        np.ndarray (1, dim) 형태의 벡터
    """
    with torch.no_grad():
        inputs = processor.process_images([image]).to(llm_model.device)
        embeddings = llm_model(**inputs)  # shape: (1, dim)
    return embeddings.cpu().numpy()
