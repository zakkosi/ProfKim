# utils/faiss_utils.py
import faiss
import numpy as np
from PIL import Image
import os
import pickle
from Utils.encoding_utils import encode_text_ada002 as encode_text

def save_faiss_index(index, path):
    faiss.write_index(index, f"{path}.index")

def load_faiss_index(path):
    return faiss.read_index(f"{path}.index")

def save_id_mapping(mapping: dict, path):
    with open(f"{path}_ids.pkl", "wb") as f:
        pickle.dump(mapping, f)

def load_id_mapping(path):
    with open(f"{path}_ids.pkl", "rb") as f:
        return pickle.load(f)

def build_faiss_index(vectors: np.ndarray, id_list: list[str], path: str):
    dim = vectors.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(vectors)
    save_faiss_index(index, path)
    save_id_mapping({i: id_ for i, id_ in enumerate(id_list)}, path)
    return index

def retrieve_similar_images_from_text(text_description: str, index_path: str, top_k: int = 1) -> list[str]:
    """
    Qwen2.5-VL이 생성한 텍스트 설명 → 텍스트 벡터 → FAISS 검색
    """
    try:
        # 1. 텍스트 벡터 인코딩
        print("[디버그] 텍스트 쿼리:", text_description)
        text_vec = encode_text(text_description)

        # 2. Tensor → numpy 변환
        if hasattr(text_vec, "cpu"):
            text_vec = text_vec.cpu().numpy()

        # 3. FAISS용 벡터 형식으로 reshape + float32
        query_vec = text_vec.reshape(1, -1).astype("float32")
        print("[디버그] 최종 쿼리 벡터 차원:", query_vec.shape)

        # 4. 인덱스 로드 및 검색
        index = load_faiss_index(index_path)
        id_map = load_id_mapping(index_path)

        scores, indices = index.search(query_vec, top_k)
        print("[디버그] Top-5 Scores:", scores)
        print("[디버그] Top-5 Indices:", indices)

        return [id_map[i] for i in indices[0]]

    except Exception as e:
        print(f"[예외] 검색 실패: {e}")
        return [f"[오류] 검색 실패: {e}"]
