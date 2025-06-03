import os
import json
from Utils.image_utils import load_image_as_bytes, load_image_resized
from Utils.faiss_utils import retrieve_similar_images_from_text
from inference.image_description import describe_image, describe_image_contextual
from inference.professor_response import generate_professor_response

# ------------------------------
# [1] 입력 준비
# ------------------------------
IMAGE_PATH = "sample_input/TEST_1.png"
QUESTION_PATH = "sample_input/question.txt"
VECTOR_DB_PATH = "vectorstore/index_textonly.faiss"

with open(QUESTION_PATH, 'r', encoding='utf-8') as f:
    question = f.read().strip()

image_bytes_qwen = load_image_as_bytes(IMAGE_PATH)
# image_resized = load_image_resized(IMAGE_PATH)(추가 스터디용용)
# ------------------------------
# [2] 이미지 설명 + 키워드 추출 (Qwen2.5)
# ------------------------------
print("[1] Qwen2.5-VL로 이미지 설명 생성 중...")
image_description = describe_image(image_bytes_qwen)
print("\n📌 이미지 설명:")
print(image_description)

# ------------------------------
# [3] 텍스트 기반 FAISS 검색
# ------------------------------
print("\n[2] FAISS로 유사 이미지 검색 중...")
retrieved_images = retrieve_similar_images_from_text(image_description, VECTOR_DB_PATH)
print("\n🔎 유사 이미지 ID 리스트:", retrieved_images)

# ------------------------------
# [4] Qwen2.5-VL로 문맥 기반 설명 (Top-K Doc)
# ------------------------------
print("\n[3] Qwen2.5-VL로 유사 이미지들에 대한 문맥 설명 생성 중...")

retrieved_descriptions = []

for doc_id in retrieved_images:
    try:
        meta_path = f"vectorstore/meta/{doc_id}.json"
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)

        text_block = meta["text"]
        page_ids = meta["pages"]

        image_bytes_list = []
        for pid in page_ids:
            path = f"converted_pages/page_{pid}.jpg"
            if os.path.exists(path):
                img_bytes = load_image_as_bytes(path)
                image_bytes_list.append(img_bytes)

        if not image_bytes_list:
            retrieved_descriptions.append(f"[{doc_id}] 이미지 없음")
            continue

        print(f"[{doc_id}] Qwen2.5-VL 문맥 호출 중...")
        response = describe_image_contextual(
            images=image_bytes_list,
            question=text_block + "\n이 내용과 시각자료를 바탕으로 강의 흐름을 설명해줘."
        )
        retrieved_descriptions.append(f"[{doc_id}]\n{response}")

    except Exception as e:
        retrieved_descriptions.append(f"[{doc_id}] 처리 실패: {e}")

context_summary = "\n\n".join(retrieved_descriptions)
print("\n🧠 문맥 요약:")
print(context_summary)

# ------------------------------
# [5] 교수님 스타일 최종 응답 (LLaMA3)
# ------------------------------
print("\n[4] LLaMA3로 최종 교수님 응답 생성 중...")
final_response = generate_professor_response(question, context_summary, retrieved_images)
print("\n🎓 교수님의 최종 조언:")
print(final_response)
