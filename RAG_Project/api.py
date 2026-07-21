from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import traceback

from rag import get_answer

app = FastAPI(
    title="Enterprise RAG Chatbot API",
    description="API for querying Bank PDF using RAG",
    version="1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    question: str
    session_id: str = "default"


class ChatResponse(BaseModel):
    answer: str


@app.get("/")
def home():
    return {
        "status": "Running",
        "message": "Enterprise RAG Chatbot API"
    }


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):

    try:

        answer = get_answer(
            request.question,
            request.session_id
        )

        return ChatResponse(answer=answer)

    except Exception as e:

        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )