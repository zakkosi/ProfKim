# inference/professor_response.py

import ollama

def generate_professor_response(question: str, image_description: str, retrieved_ids: list[str]) -> str:
    """
    LLaMA3를 이용해 교수님 스타일의 학습 조언 생성

    Args:
        question: 사용자의 질문
        image_description: Qwen2.5가 분석한 이미지 설명
        retrieved_ids: ColPali가 찾아낸 유사 이미지 ID 리스트 (ex: page_12.jpg 등)

    Returns:
        답변 텍스트 (str)
    """

    # 1. 유사 이미지 결과를 텍스트로 정리
    related_section_text = "\n".join([f"- 관련 문서: {rid}" for rid in retrieved_ids])

    # 2. 프롬프트 구성
    prompt = f"""
당신은 대학 강의의 교수님입니다.
학생의 질문, 이미지 분석 결과, 관련 강의자료를 참고하여,
심도 있는 학습 피드백을 제공해주세요.

[학생 질문]
{question}

[학생이 제공한 이미지 분석 결과]
{image_description}

[강의자료 유사 페이지 목록]
{related_section_text}

[교수님의 조언]
"""

    # 3. LLaMA3 호출 (Ollama)
    try:
        response = ollama.chat(
            model='llama3:8b',
            messages=[{"role": "user", "content": prompt}]
        )
        return response['message']['content']
    except Exception as e:
        return f"[오류] LLaMA3 응답 실패: {e}"
