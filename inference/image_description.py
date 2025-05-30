# image_description.py

from PIL import Image
import io
import ollama

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
        return response['message']['content']
    except Exception as e:
        return f"[오류] Qwen2.5-VL 처리 실패: {e}"
