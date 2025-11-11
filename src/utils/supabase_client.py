import os
from supabase import create_client, Client

class SupabaseClient:
    """Thin wrapper quanh Supabase Python SDK dành cho Vector Search."""
    def __init__(self) -> None:
        url: str = os.environ["SUPABASE_URL"]
        key: str = os.environ["SUPABASE_SERVICE_KEY"]  # service_role key
        self.client: Client = create_client(url, key)

    # ---------- VECTOR SEARCH ----------
    def search_vectors(
        self,
        table: str,
        query_embedding: list[float],
        filters: dict | None = None,
        top_k: int = 5,
    ) -> list[dict]:
        """
        Gọi RPC `match_fts` (Supabase vector extension) trả về top-k rows.
        Bạn phải tạo hàm RPC này trong SQL migrations → docs của Supabase.
        """
        payload = {
            "query_embedding": query_embedding,
            "match_count": top_k,
            **(filters or {}),
        }
        resp = (
            self.client.rpc("match_fts", payload)  # type: ignore
            .execute()
        )
        if resp.error:
            raise RuntimeError(resp.error.message)
        return resp.data  # type: ignore

    # ---------- CRUD (nếu cần) ----------
    def insert(self, table: str, rows: list[dict]) -> None:
        resp = self.client.table(table).insert(rows).execute()
        if resp.error:
            raise RuntimeError(resp.error.message)