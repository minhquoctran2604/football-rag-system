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

    # ---------- AUTHENTICATION ----------
    def sign_up(self, email: str, password: str) -> dict:
        """Register a new user with email and password."""
        try:
            response = self.client.auth.sign_up({
                "email": email,
                "password": password
            })
            return {
                "user": response.user,
                "session": response.session
            }
        except Exception as e:
            raise RuntimeError(f"Sign up failed: {str(e)}")

    def sign_in(self, email: str, password: str) -> dict:
        """Sign in an existing user with email and password."""
        try:
            response = self.client.auth.sign_in_with_password({
                "email": email,
                "password": password
            })
            return {
                "user": response.user,
                "session": response.session,
                "access_token": response.session.access_token if response.session else None
            }
        except Exception as e:
            raise RuntimeError(f"Sign in failed: {str(e)}")

    def sign_out(self, access_token: str) -> None:
        """Sign out the current user."""
        try:
            self.client.auth.sign_out()
        except Exception as e:
            raise RuntimeError(f"Sign out failed: {str(e)}")

    def get_user(self, access_token: str) -> dict:
        """Get the current user information from access token."""
        try:
            response = self.client.auth.get_user(access_token)
            return response.user
        except Exception as e:
            raise RuntimeError(f"Get user failed: {str(e)}")