from fastapi import FastAPI,APIRouter
from app.router.chatbot import chatbot_router
from app.router.text_file import text_router
app = FastAPI()

@app.get("/")
def root_index():
    return "서버실행중"

user_router = APIRouter(prefix="/users")
img_processing_router = APIRouter(prefix="/image")

@user_router.get("/sign",tags=['Users'])
def sign_index():
    return "회원가입입니다."




app.include_router(user_router)
app.include_router(chatbot_router)
app.include_router(img_processing_router)
app.include_router(text_router)
