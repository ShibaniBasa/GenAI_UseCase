from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import traceback

# Import your RAG function
# Make sure rag.py contains a function named get_answer()
from rag import get_answer

# --------------------------------------------------
# FastAPI App
# --------------------------------------------------

app = FastAPI(
    title="Enterprise RAG Chatbot API",
    description="API for querying Bank PDF using RAG",
    version="1.0"
)

# --------------------------------------------------
# Enable CORS
# --------------------------------------------------

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],      # Change in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------
# Request Model
# --------------------------------------------------

class ChatRequest(BaseModel):
    question: str

# --------------------------------------------------
# Response Model
# --------------------------------------------------

class ChatResponse(BaseModel):
    answer: str

# --------------------------------------------------
# Health Check
# --------------------------------------------------

@app.get("/")
def home():
    return {
        "status": "Running",
        "message": "Enterprise RAG Chatbot API"
    }

# --------------------------------------------------
# Chat Endpoint
# --------------------------------------------------

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):

    try:

        question = request.question.strip()

        if not question:
            raise HTTPException(
                status_code=400,
                detail="Question cannot be empty."
            )

        answer = get_answer(question)
        print(f"Question: {question} | Answer: {answer}")
        return ChatResponse(answer=answer)

    except Exception as e:

        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )