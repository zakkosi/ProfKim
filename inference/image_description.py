# image_description.py

from PIL import Image
import io
import ollama
from konlpy.tag import Okt

def extract_keywords(text: str) -> str:
    okt = Okt()
    nouns = okt.nouns(text)
    return " ".join(sorted(set(nouns)))

def describe_image(image_bytes: bytes, question: str = "이 이미지를 설명해줘.") -> str:
    try:
        response = ollama.chat(
            model='qwen2.5vl:7b',
            messages=[
                {
                    'role': 'user',
                    'content': question,
                    'images': [image_bytes]
                }
            ]
        )
        full_desc = response['message']['content']
        keywords = extract_keywords(full_desc)

        # ✅ 키워드 + 설명을 함께 리턴
        return f"[키워드 요약]: {keywords}\n\n{full_desc}"
    
    except Exception as e:
        return f"[오류] Qwen2.5-VL 처리 실패: {e}"

def describe_image_contextual(images: list, question: str) -> str:
    """
    Qwen2.5-VL로 여러 이미지와 함께 텍스트 기반 문맥 설명 생성
    """
    try:
        response = ollama.chat(
            model='qwen2.5vl:7b',
            messages=[{
                'role': 'user',
                'content': question,
                'images': images
            }]
        )
        return response['message']['content']
    except Exception as e:
        return f"[오류] Qwen2.5-VL 처리 실패: {e}"