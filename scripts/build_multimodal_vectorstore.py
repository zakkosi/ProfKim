# build_textonly_vectorstore.py

import os
import pickle
from tqdm import tqdm
import numpy as np
from Utils.faiss_utils import build_faiss_index
from Utils.encoding_utils import encode_text_ada002, generate_sliding_windows

# 멀티모달 데이터 경로 설정
IMAGE_DIR = "./converted_pages"
TEXT_JSON = "./vectorstore/text_per_page.json"
META_SAVE_PATH = "./vectorstore/meta"
os.makedirs(META_SAVE_PATH, exist_ok=True)

# FAISS용 벡터 저장 리스트
text_vectors = []
image_vectors = []
metadata_list = []
query_vectors = []  # ✅ 쿼리 시뮬레이션용 멀티모달 벡터

# 슬라이딩 윈도우 생성 (3페이지 단위)
windows = generate_sliding_windows(IMAGE_DIR, TEXT_JSON, window_size=3)

for entry in tqdm(windows):
    doc_id = entry["doc_id"]
    text = entry["text"]

    try:
        # 텍스트 벡터 (ada-002)
        text_vec = encode_text_ada002(text)
        text_vectors.append((doc_id, text_vec))

        # 메타 정보 저장
        meta = {
            "doc_id": doc_id,
            "text": text,
            "image_path": entry.get("image_path", None),
            "pages": entry.get("pages", [])
        }
        metadata_list.append(meta)
        with open(os.path.join(META_SAVE_PATH, f"{doc_id}.json"), 'w', encoding='utf-8') as mf:
            import json
            json.dump(meta, mf, ensure_ascii=False, indent=2)

    except Exception as e:
        print(f"[오류] {doc_id} 처리 실패: {e}")

# 텍스트 벡터 저장
with open("./vectorstore/text_vectors.pkl", "wb") as f:
    pickle.dump(text_vectors, f)

print("✅ 텍스트 벡터 저장 완료")

# FAISS 인덱스 생성 (4096차원 기준)
text_vectors.sort(key=lambda x: x[0])
doc_ids = [doc_id for doc_id, _ in text_vectors]
text_arr = np.stack([vec for _, vec in text_vectors])

build_faiss_index(text_arr, doc_ids, "vectorstore/index_textonly.faiss")
print("✅ 텍스트 전용 FAISS 인덱스 저장 완료")
