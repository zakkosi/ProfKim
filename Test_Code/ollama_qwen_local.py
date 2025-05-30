import ollama
from PIL import Image
import io

# 1. 입력 이미지 및 질문 경로
image_path = "sample_input/image.png"
question_path = "sample_input/question.txt"

# 2. 이미지 로딩 및 byte로 변환
image = Image.open(image_path).convert("RGB")
img_byte_arr = io.BytesIO()
image.save(img_byte_arr, format="PNG")
image_bytes = img_byte_arr.getvalue()

# 3. 질문 텍스트 로딩
with open(question_path, "r", encoding="utf-8") as f:
    question = f.read().strip()

# 4. Ollama API 호출 - 첫 번째 응답 (이미지 설명)
print(" 질의:", question)
print(" 이미지 분석 중...")

try:
    res1 = ollama.chat(
        model='qwen2.5vl:7b',
        messages=[
            {
                'role': 'user',
                'content': question,
                'images': [image_bytes]
            }
        ]
    )
    image_analysis = res1['message']['content']
    print("\n 이미지 설명:")
    print(image_analysis)
except Exception as e:
    print(f" 오류 발생 (이미지 설명): {e}")
    exit()

# 5. Ollama API 호출 - 두 번째 응답 (추가 학습 조언)
print("\n 추가 설명 요청 중...")

try:
    res2 = ollama.chat(
        model='llama3:8b',
        messages=[
            {
                'role': 'user',
                'content': f"당신은 교수님입니다. 다음 이미지 분석 내용을 기반으로 학술적으로 유의할 점을 알려줘:\n\n{image_analysis}"
            }
        ]
    )
    print("\n💡 Professor 한마디:")
    print(res2['message']['content'])

except Exception as e:
    print(f" 오류 발생 (학습 조언): {e}")
