from typing import Any
from src.utils.supabase_client import SupabaseClient
from src.utils.gemini_client import GeminiClient
from src.rag.types import QueryContext,Strategy
class Retriever:
    """Vector + metadata retrieval."""
    def __init__(self, supabase: SupabaseClient, gemini_client: GeminiClient ):
        self.supabase = supabase
        self.gemini = gemini_client

    def llm_select_table(self, user_question) -> str:
        prompt = f"""Given the {user_question}, select the most relevant table from the following options: players, teams. Only return the table name. """
        
        response = self.gemini.chat(
            system_prompt="You are an expert database assistant.",
            user_prompt=prompt
        ).lower()
        
        if "team" in response or "teams" in response or "club" in response or "clubs" in response:
            return "teams"
        
        if "player" in response or "players" in response or "footballer" in response or "footballers" in response:
            return "players"

    def retrieve_by_filters(self, query: str, filters: dict | None = None, top_k: int = 5):
        table = self.llm_select_table(query)
        return self.supabase.search_by_filters(
            table=table,
            filters=filters or {},
            top_k=top_k,
        )

    def retrieve_semantic(self, query: str, query_embedding: list[float], top_k: int = 5):
        table = self.llm_select_table(query)
        return self.supabase.search_vectors(
            table=table,
            query_embedding=query_embedding,
            filters=None,
            top_k=top_k,
        )

    def retrieve_hybrid(self, query: str, query_embedding: list[float], filters: dict | None = None, top_k: int = 5):
        table = self.llm_select_table(query)
        return self.supabase.search_vectors(
            table=table,
            query_embedding=query_embedding,
            filters=filters,
            top_k=top_k,
        )

    def __call__(   
        self,
        query: str,
        query_embedding: list[float],
        filters: dict | None = None, 
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        return self.supabase.search_vectors(
            table=self.llm_select_table(query),
            query_embedding=query_embedding,
            filters=filters,
            top_k=top_k,
        )
        