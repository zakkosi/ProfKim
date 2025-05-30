import os
import glob
from pdf2image import convert_from_path
from PIL import Image
import torch
from transformers.utils.import_utils import is_flash_attn_2_available
from colpali_engine.models import ColQwen2, ColQwen2Processor
from byaldi import RAGMultiModalModel

# ------------------------
# 1. 모델 로딩
# ------------------------
print("모델 로딩 중...")

# 질의응답용 ColQwen2
llm_model = ColQwen2.from_pretrained(
    "vidore/colqwen2-v1.0",
    torch_dtype=torch.bfloat16,
    device_map="cuda:0",
    attn_implementation="flash_attention_2" if is_flash_attn_2_available() else None,
).eval()

processor = ColQwen2Processor.from_pretrained("vidore/colqwen2-v1.0")

# 검색용 RAG 인덱서
rag_model = RAGMultiModalModel.from_pretrained("vidore/colqwen2-v1.0")

# ------------------------
# 2. PDF → 이미지 변환
# ------------------------
pdf_path = "test_materials/example.pdf"
image_dir = "converted_pages"
os.makedirs(image_dir, exist_ok=True)

image_paths = sorted(glob.glob(f"{image_dir}/page_*.jpg"))
if not image_paths:
    print("PDF → 이미지 변환 중...")
    images = convert_from_path(pdf_path)
    for idx, img in enumerate(images):
        save_path = f"{image_dir}/page_{idx}.jpg"
        img.save(save_path, "JPEG")
        image_paths.append(save_path)
    print(f"{len(image_paths)}개 이미지 추출 완료")
else:
    print("기존 이미지 파일이 존재함. 변환 건너뜀.")

# ------------------------
# 3. RAG 인덱싱
# ------------------------
print("🔍 이미지 인덱싱 중...")
rag_model.index("converted_pages", index_name="default", overwrite=True)

# ------------------------
# 4. 질의 및 검색
# ------------------------
question = "이 문서에서 캥거루는 어디에 있나요?"
print(f"\n질의: {question}")

top_docs = rag_model.search(question) 
print(top_docs)
retrieved_images = [
    Image.open(f"converted_pages/page_{doc['doc_id']}.jpg").convert("RGB")
    for doc in top_docs
]
image_batch = processor.process_images(retrieved_images).to(llm_model.device)
query_batch = processor.process_queries([question]).to(llm_model.device)

# ------------------------
# 5. LLM으로 답변 생성
# ------------------------
with torch.no_grad():
    image_embeds = llm_model(**image_batch)
    query_embeds = llm_model(**query_batch)

answer = processor.score_multi_vector(query_embeds, image_embeds)

print("\n✅ 응답 결과:")
print(answer)

scores = answer[0].tolist()  # 텐서를 리스트로 변환

print("\n 유사한 페이지 및 점수:")
for doc, score in zip(top_docs, scores):
    page = doc['doc_id']  # doc_id는 이미지 번호(page_32.jpg 같은)
    print(f" page_{page}.jpg → 유사도 점수: {score:.4f}")
