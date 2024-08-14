from fastapi import APIRouter
from langchain.prompts import PromptTemplate,ChatPromptTemplate,MessagesPlaceholder
from langchain_community.chat_models import ChatOllama
from pydantic import BaseModel
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
#from langchain_community.document_loaders import 
from langchain.memory import ConversationBufferMemory
from langchain_huggingface.embeddings import HuggingFaceEndpointEmbeddings
from langchain_community.embeddings import OllamaEmbeddings
OPENAI_DB_PATH = "./openai_db"

#emb = HuggingFaceEndpointEmbeddings()
#llama_emb = OllamaEmbeddings(model="mxbai-embed-large",# 임베딩 모델 지정(ollama 홈페이지에서 찾음)num_gpu=1,show_progress=True)
emb = OpenAIEmbeddings()
class Prompt(BaseModel):
    text: str

prompt_dic = {}

model = ChatOllama(model="llama3.1:latest")

prompt_dic['wether'] = "{text}의 날씨를 알려줘"
prompt_dic['books'] = "{text}라는 책에 주인공을 알려줘"
prompt_dic['system'] = '너는 {text}로 말하는 ai야'

template_vec = """
    아래의 책에 대한 내용은 검색된 Context 기반으로 Question 에 한국어로 대답을 하도록 해
    Question : {text} 
    Context: {context} 
    Answer:
"""
system_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system","""너는 영문의 책은 한국어로 번역하고 책의 내용을 문단 별로 읽어 주는 어시스턴트야.
            그리고 책의 주인공에 대해서 설명해주고 등장인물의 성격을 분석하며 등장인물들 사이에 대화에서 감정을 분석하는 어시스턴트야.
            처음 시작은 책의 주인공에 행적을 분석해서 성격을 분석해서 한마디로 출력해줘. 
            이후에는 사용자가 입력하는 chapter 부터 책의 내용을 문단별로 출력해줘.
            '>'가 입력되면 책의 다음 문단을 출력해줘. """
        ),
        MessagesPlaceholder(variable_name="history"),# 이전 대화 내용
        (
            "human","{input}"
        )
    ]
)
memory_store = {}

chatbot_router = APIRouter(prefix="/chatbot")

def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])#각각의 문서 내용을 줄바꿈으로 구분된 하나의 문자열로 반환하는 역할을 합니다.


@chatbot_router.get("/llama31",tags=["Chatbot"])
def chatbot_models():
    return "챗봇 대화입니다."

@chatbot_router.get("/mistral",tags=['Chatbot'])
def chatbot_models2():
    return "미스트랄입니다."

@chatbot_router.post("/prompt",tags=['Chatbot'])
def chatbot_model3(key : str,prompt_model : Prompt):
    prompt = PromptTemplate.from_template(template=prompt_dic[key])
    chain = prompt|model
    answer = chain.invoke(prompt_model)
    return answer.content,prompt_model.text

#벡터화 db 불러와서 답변 생성하는 챗봇

@chatbot_router.post("/prompt_from_vec",tags=['Chatbot'])
def chatbot_model4(user_id : str, prompt_model: Prompt):

    if user_id not in memory_store:
        memory_store[user_id] = ConversationBufferMemory()

    memory = memory_store[user_id]

    openai_db = Chroma(
        persist_directory=OPENAI_DB_PATH,
        embedding_function=emb
    )
    openai_retriever = openai_db.as_retriever(
        search_kwargs={"k":5}
    )
    conversation_history = memory.load_memory_variables({})

    


    context = openai_retriever.invoke(prompt_model.text)
    custom_prompt = ChatPromptTemplate.from_template(template_vec)
    # 필요한 모든 변수를 템플릿에 전달
    formatted_prompt = custom_prompt.format(
        history=conversation_history,
        context=context,
        text=prompt_model.text
    )
    rag_chain = (
        {"context":openai_retriever | format_docs,"text":RunnablePassthrough()}
         | system_prompt
         | custom_prompt
         | model
         | StrOutputParser()
    )
    response = rag_chain.invoke(formatted_prompt)
    #이전 대화를 기억하는 부분 만들기

    # 현재 대화와 AI 응답을 메모리에 추가
    memory.save_context({"user": prompt_model.text}, {"ai": response})

    return response