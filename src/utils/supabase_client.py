import os
from supabase import create_client, Client

class SupabaseClient:
    def __init__(self) -> None:
        url: str = os.environ["SUPABASE_URL"]
        key: str = os.environ["SUPABASE_SERVICE_KEY"]  
        self.client: Client = create_client(url, key)

    
    def search_vectors(
        self,
        table: str,
        query_embedding: list[float],
        filters: dict | None = None,
        top_k: int = 5,
    ) -> list[dict]:
        payload = {
            "query_embedding": query_embedding,
            "match_count": top_k,
            **(filters or {}),
        }
        resp = self.client.rpc("match_vector", payload).execute() 
        if resp.error:
            raise RuntimeError(resp.error.message)
        return resp.data  # type: ignore

    # CRUD
    def insert(self, table: str, rows: list[dict]) -> None:
        resp = self.client.table(table).insert(rows).execute()
        if resp.error:
            raise RuntimeError(resp.error.message)