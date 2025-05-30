import fitz  # PyMuPDF
import json
import os

def extract_text_per_page(pdf_path, output_json_path):
    doc = fitz.open(pdf_path)
    text_dict = {}

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()
        text_dict[str(page_num + 1)] = text.strip()

    with open(output_json_path, 'w', encoding='utf-8') as f:
        json.dump(text_dict, f, ensure_ascii=False, indent=2)

    print(f"[완료] 텍스트 저장됨 → {output_json_path}")

if __name__ == "__main__":
    pdf_path = "./input_pdf/stanford_lecture.pdf"
    output_json = "./vectorstore/text_per_page.json"

    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    extract_text_per_page(pdf_path, output_json)
