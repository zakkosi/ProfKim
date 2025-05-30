from utils.image_utils import resize_image
from utils.faiss_utils import load_faiss_index, load_id_mapping
from models.colpali_model import encode_image
import numpy as np
from PIL import Image

def retrieve_similar_images(image: Image.Image, index_path: str, top_k: int = 5) -> list[str]:
    """
    ColPali로 이미지 임베딩 → FAISS 검색 → 유사 이미지 ID 리스트 반환
    """
    try:
        # 1. 이미지 리사이징
        image_resized = resize_image(image, size=(224, 224))

        # 2. ColPali 임베딩 추출
        image_vec = encode_image(image_resized)  # shape: (1, dim) np.ndarray

        # 3. FAISS 인덱스 로딩
        index = load_faiss_index(index_path)
        id_map = load_id_mapping(index_path)

        # 4. 유사 이미지 검색
        scores, indices = index.search(image_vec, top_k)
        return [id_map[i] for i in indices[0]]
    except Exception as e:
        return [f"[오류] 유사 이미지 검색 실패: {e}"]
