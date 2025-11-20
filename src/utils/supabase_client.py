import os
from supabase import create_client, Client

class SupabaseClient:
    def __init__(self) -> None:
        url: str = os.environ["SUPABASE_URL"]
        key: str = os.environ["SUPABASE_SERVICE_KEY"]
        self.client = create_client(url, key)

    def search_vectors(
        self,
        table: str,
        query_embedding: list[float],
        filters: dict = None,
        top_k: int = 5,
    ) -> list[dict]:
        """Search vectors using embedding"""
        
        # Select RPC based on table
        rpc_name = "match_teams" if table == "teams" else "match_players"
        
        payload = {
            "query_embedding": query_embedding,
            "match_count": top_k,
            "match_threshold": 0.3, # Default threshold
            "filter": filters or {},
        }
        
        try:
            resp = self.client.rpc(rpc_name, payload).execute()
            return resp.data
        except Exception as e:
            print(f"Error calling RPC {rpc_name}: {e}")
            return []

    def insert(self, table: str, rows: list[dict]) -> list[dict]:
        """Insert rows into table"""
        resp = self.client.table(table).insert(rows).execute()
        return resp.data

    def upsert(self, table: str, rows: list[dict]) -> list[dict]:
        """Upsert rows into table"""
        resp = self.client.table(table).upsert(rows).execute()
        return resp.data

    def search_by_filters(
        self, table: str, filters: dict, top_k: int = 5
    ) -> list[dict]:
        """Search by filters"""
        query = self.client.table(table).select("*").limit(top_k)
        for key, value in filters.items():
            query = query.eq(key, value)
        resp = query.execute()
        return resp.data

    def table(self, table_name: str):
        """Get table reference"""
        return self.client.table(table_name)
