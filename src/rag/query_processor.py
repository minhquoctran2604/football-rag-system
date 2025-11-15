from src.utils.gemini_client import GeminiClient


class QueryProcessor:
    """Tạo embedding + (tùy chọn) phân tích intent/filters."""
    def __init__(self, gemini_client: GeminiClient):
        self.gemini = gemini_client

    def __call__(self, query: str) -> dict:
        embedding = self.gemini.get_embedding(query)
        return {"embedding": embedding, "filters": None}