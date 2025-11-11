from typing import Any
from src.utils.supabase_client import SupabaseClient


class Retriever:
    """Vector + metadata retrieval."""
    def __init__(self, supabase: SupabaseClient, table: str = "documents"):
        self.supabase = supabase
        self.table = table

    def __call__(
        self,
        query_embedding: list[float],
        filters: dict | None = None,
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        return self.supabase.search_vectors(
            table=self.table,
            query_embedding=query_embedding,
            filters=filters,
            top_k=top_k,
        )