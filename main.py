import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

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


# ----- routes -----
@app.post("/ask", response_model=RAGResponse)
def ask(question: Question):
    try:
        result = pipeline(question.query)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("main:app", reload=True)