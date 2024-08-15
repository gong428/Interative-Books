import gradio as gr
from transformers import AutoTokenizer, AutoModel
import torch

# 임베딩 생성을 위한 모델과 토크나이저 로드
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")

def embed_text(text):
    # 텍스트를 토크나이징
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        # 모델을 통해 임베딩 생성
        outputs = model(**inputs)
        # 마지막 은닉 상태의 평균을 취해 임베딩 벡터로 사용
        embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().tolist()
    return embeddings

def process_file(file):
    # 파일 내용 읽기
    text = file.read().decode("utf-8")
    # 임베딩 생성
    embeddings = embed_text(text)
    return embeddings

# Gradio 인터페이스 설정
iface = gr.Interface(
    fn=process_file,  # 파일 처리 함수
    inputs=gr.inputs.File(),  # 파일 입력
    outputs="json",  # 출력은 JSON 형태
    title="File to Embedding",
    description="Upload a text file and get its vector embedding."
)

# 인터페이스 실행
if __name__ == "__main__":
    iface.launch()
