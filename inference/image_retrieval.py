from Utils.faiss_utils import load_faiss_index, load_id_mapping
from build_multimodal_vectorstore import encode_text  # 동일한 인코더 재사용
import numpy as np

def retrieve_similar_images_from_text(text_description: str, index_path: str, top_k: int = 5) -> list[str]:
    """
    Qwen2.5-VL이 생성한 텍스트 설명 → 텍스트 벡터 → FAISS 검색
    """
    try:
        # 1. 텍스트 벡터 인코딩
        text_vec = encode_text(text_description)  # shape: (1, 4096)

        # 2. 이미지 벡터는 1024, 텍스트는 4096 → 전체 5120차원
        dummy_img = np.zeros((1, 1024))  # 이미지 임베딩은 사용 안 함
        query_vec = np.concatenate([dummy_img, text_vec], axis=-1)

        # 3. 인덱스 로드 및 검색
        index = load_faiss_index(index_path)
        id_map = load_id_mapping(index_path)

        scores, indices = index.search(query_vec.astype("float32"), top_k)
        return [id_map[i] for i in indices[0]]
    except Exception as e:
        return [f"[오류] 검색 실패: {e}"]
