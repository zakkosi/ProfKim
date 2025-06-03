import os
from inference.image_description import describe_image
from inference.professor_response import generate_professor_response
from Utils.faiss_utils import retrieve_similar_images_from_text

# ------------------------------
# 설정
# ------------------------------
IMAGE_PATH = "sample_input/image.jpg"
QUESTION_PATH = "sample_input/question.txt"
VECTOR_DB_PATH = "vectorstore/index.faiss"

# ------------------------------
# 1. 입력 준비
# ------------------------------
with open(QUESTION_PATH, 'r', encoding='utf-8') as f:
    question = f.read().strip()

image_bytes_qwen = load_image_as_bytes(IMAGE_PATH)
image_resized = load_image_resized(IMAGE_PATH)

# ------------------------------
# 2. 이미지 설명 생성 (Qwen2.5-VL)
# ------------------------------
print("[1] Qwen2.5-VL로 이미지 설명 생성 중...")
image_description = describe_image(image_bytes_qwen)
print("\n 이미지 설명:")
print(image_description)

# ------------------------------
# 3. 유사 이미지 검색 (ColPali + FAISS)
# ------------------------------
print("\n[2] ColPali로 유사 이미지 검색 중...")
retrieved_images = retrieve_similar_images_from_text(image_description, VECTOR_DB_PATH)
print("\n 유사 이미지 인덱스:", retrieved_images)

# ------------------------------
# 4. 교수 스타일 응답 생성 (LLM)
# ------------------------------
print("\n[3] 교수 스타일 학습 조언 생성 중...")
professor_msg = generate_professor_response(question, image_description, retrieved_images)
print("\n🎓 교수님 한마디:")
print(professor_msg)
