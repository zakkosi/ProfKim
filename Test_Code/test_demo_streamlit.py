import streamlit as st
import ollama
from PIL import Image
import io

def get_image_details(image_bytes):

    try:
        res = ollama.chat(
            model='qwen2.5vl:7b',  
            messages=[
                {
                    'role': 'user',
                    'content': '이 이미지를 분석해서 학술적으로 설명해줘줘.',
                    'images': [image_bytes]
                }
            ]
        )
        return res['message']['content']
    except Exception as e:
        return f"오류 발생: {e}"

st.set_page_config(layout="wide")
st.title("TEST")
st.markdown("이미지를 업로드하면 내용을 분석해 드립니다.")

uploaded_file = st.file_uploader("이미지 업로드", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="업로드된 이미지지", use_column_width=True)

    # 이미지를 바이트로 변환
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format=image.format)
    img_byte_arr = img_byte_arr.getvalue()

    st.markdown("---")
    st.subheader("이미지지 분석 결과:")

    with st.spinner("이미지를를 분석 중입니다..."):
        receipt_details = get_image_details(img_byte_arr)
        st.text_area("분석 내용", receipt_details, height=300)

   
        if "오류 발생" not in receipt_details:
            st.markdown("---")
            st.subheader("Professor 한마디 💬")
            # Qwen 모델에 다시 요청하여 회계 처리 관련 조언을 얻을 수 있습니다.
            # 여기서는 간단한 메시지를 표시합니다.
            try:
                accounting_advice_res = ollama.chat(
                    model='qwen2:7b-chat-v1.5-q6_K', # 사용자의 모델명으로 수정 필요
                    messages=[
                        {
                            'role': 'user',
                            'content': f"다음 이미지지 분석 결과를 바탕으로 학습습 시 유의사항을 알려줘: {receipt_details}"
                        }
                    ]
                )
                accounting_advice = accounting_advice_res['message']['content']
                st.info(accounting_advice)
            except Exception as e:
                st.warning(f"학습 조언을 가져오는 중 오류 발생: {e}")
        else:
            st.error("학습 분석에 실패하여 회계 조언을 제공할 수 없습니다.")

else:
    st.info("왼쪽 사이드바에서 이미지를 업로드해주세요.")
