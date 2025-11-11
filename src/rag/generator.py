from src.utils.gemini_client import GeminiClient


class ResponseGenerator:
    """Sinh câu trả lời cuối cùng từ context."""
    def __init__(self, openai_client: GeminiClient):
        self.openai = openai_client
        self.system_prompt = (
            "You are a helpful football analytics assistant. "
            "Answer in Vietnamese and cite nguồn nếu có."
        )

    def __call__(self, query: str, context: list[dict]) -> str:
        context_txt = "".join(d["content"] for d in context)
        user_prompt = (
            f"Context:{context_txt}"
            f"Question: {query}"
            "Trả lời ngắn gọn."
        )
        return self.openai.chat(self.system_prompt, user_prompt)