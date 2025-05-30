# build_multimodal_vectorstore.py

import os
import pickle
from tqdm import tqdm
import numpy as np
from PIL import Image

from Utils.image_utils import load_image_as_tensor
from Utils.encoding_utils import encode_image_clip, encode_text_ada002, generate_sliding_windows

# 멀티모달 데이터 경로 설정
IMAGE_DIR = "./converted_pages"
TEXT_JSON = "./vectorstore/text_per_page.json"
META_SAVE_PATH = "./vectorstore/meta"
os.makedirs(META_SAVE_PATH, exist_ok=True)

# FAISS용 벡터 저장 리스트
text_vectors = []
image_vectors = []
metadata_list = []

# 슬라이딩 윈도우 생성
windows = generate_sliding_windows(IMAGE_DIR, TEXT_JSON, window_size=3)  # 리스트 반환

for entry in tqdm(windows):
    doc_id = entry["doc_id"]
    image_path = entry["image_path"]
    full_image_path = os.path.join(IMAGE_DIR, image_path)
    text = entry["text"]

    try:
        # 텍스트 벡터 (ada-002)
        text_vec = encode_text_ada002(text)
        text_vectors.append((doc_id, text_vec))

        # 이미지 벡터 (CLIP)
        image_tensor = load_image_as_tensor(full_image_path)
        image_vec = encode_image_clip(image_tensor)
        image_vectors.append((doc_id, image_vec))

        # 메타 정보 저장
        meta = {
            "doc_id": doc_id,
            "text": text,
            "image_path": image_path,
            "pages": entry.get("pages", [])
        }
        with open(os.path.join(META_SAVE_PATH, f"{doc_id}.json"), 'w', encoding='utf-8') as mf:
            import json
            json.dump(meta, mf, ensure_ascii=False, indent=2)

    except Exception as e:
        print(f"[오류] {doc_id} 처리 실패: {e}")

# 벡터 저장
with open("./vectorstore/text_vectors.pkl", "wb") as f:
    pickle.dump(text_vectors, f)

with open("./vectorstore/image_vectors.pkl", "wb") as f:
    pickle.dump(image_vectors, f)

print("✅ 멀티모달 벡터 저장 완료")
