# src/utils/gemini_client.py
import os
import google.generativeai as genai

class GeminiClient:
    def __init__(self):
        genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        self.embed_model_name = "models/text-embedding-004"  
        self.chat_model = genai.GenerativeModel('gemini-1.5-flash')
    
    def get_embedding(self, text: str) -> list[float]:
        result = genai.embed_content(
            model=self.embed_model_name,
            content=text,
            task_type="retrieval_query"
        )
        return result['embedding']
    
    # Chat
    def chat(self, system_prompt: str, user_prompt: str) -> str:
        full_prompt = f"{system_prompt}User: {user_prompt}"
        response = self.chat_model.generate_content(full_prompt)
        return response.text