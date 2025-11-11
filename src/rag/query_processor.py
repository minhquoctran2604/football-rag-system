from src.utils.gemini_client import GeminiClient


class QueryProcessor:
    """Tạo embedding + (tùy chọn) phân tích intent/filters."""
    def __init__(self, openai_client: GeminiClient):
        self.openai = openai_client

    def __call__(self, query: str) -> dict:
        embedding = self.openai.get_embedding(query)
        # Demo: chưa parse intent/filters → để rỗng
        return {"embedding": embedding, "filters": None}