import pytest
from src.utils.gemini_client import GeminiClient
from src.rag.query_processor import QueryProcessor


def test_embedding_shape():
    openai = GeminiClient()
    qp = QueryProcessor(openai)
    out = qp("Who is the top scorer?")
    assert len(out["embedding"]) == 768