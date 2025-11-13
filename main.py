import uvicorn
from fastapi import FastAPI, HTTPException, Depends, Header
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, EmailStr
from typing import Optional

from src.utils.supabase_client import SupabaseClient
from src.utils.gemini_client import GeminiClient
from src.rag.retriever import Retriever
from src.rag.generator import ResponseGenerator
from src.rag.query_processor import QueryProcessor
from src.rag.rag_pipeline import RAGPipeline
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv()) 

app = FastAPI(title="Football-RAG")

# ----- wiring -----
supabase = SupabaseClient()
gemini_client = GeminiClient()
security = HTTPBearer()

pipeline = RAGPipeline(
    retriever=Retriever(supabase),
    generator=ResponseGenerator(gemini_client),
    query_processor=QueryProcessor(gemini_client),
)


# ----- request / response models -----
class Question(BaseModel):
    query: str


class RAGResponse(BaseModel):
    answer: str
    context: list[dict]


class SignUpRequest(BaseModel):
    email: EmailStr
    password: str


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class AuthResponse(BaseModel):
    access_token: str
    user: dict


# ----- authentication dependency -----
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Verify the access token and return the current user."""
    try:
        access_token = credentials.credentials
        user = supabase.get_user(access_token)
        return user
    except Exception as e:
        raise HTTPException(status_code=401, detail="Invalid or expired token")


# ----- routes -----
@app.post("/signup", response_model=AuthResponse)
def signup(request: SignUpRequest):
    """Register a new user."""
    try:
        result = supabase.sign_up(request.email, request.password)
        if not result.get("session"):
            raise HTTPException(status_code=400, detail="Sign up failed. Please check your email to verify your account.")
        return {
            "access_token": result["session"].access_token,
            "user": result["user"].__dict__ if hasattr(result["user"], '__dict__') else result["user"]
        }
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/login", response_model=AuthResponse)
def login(request: LoginRequest):
    """Sign in an existing user."""
    try:
        result = supabase.sign_in(request.email, request.password)
        return {
            "access_token": result["access_token"],
            "user": result["user"].__dict__ if hasattr(result["user"], '__dict__') else result["user"]
        }
    except RuntimeError as e:
        raise HTTPException(status_code=401, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/logout")
def logout(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """Sign out the current user."""
    try:
        access_token = credentials.credentials
        supabase.sign_out(access_token)
        return {"message": "Successfully logged out"}
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask", response_model=RAGResponse)
def ask(question: Question, user: dict = Depends(get_current_user)):
    """Ask a question to the RAG system (requires authentication)."""
    try:
        result = pipeline(question.query)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("main:app", reload=True)