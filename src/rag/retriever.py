from typing import Any
from src.utils.supabase_client import SupabaseClient
from src.utils.gemini_client import GeminiClient
from src.rag.types import QueryContext,Strategy
class Retriever:    
    def __init__(self, supabase: SupabaseClient, gemini_client: GeminiClient, embedding_client: Any):
        self.supabase = supabase
        self.gemini = gemini_client
        self.embedding_client = embedding_client

    def llm_select_table(self, user_question) -> str:
        prompt = f"""Given the question: "{user_question}", select the most relevant table:
        - "players" (for questions about footballers, stats, bio)
        - "teams" (for questions about clubs, stadiums, history)
        - "both" (if question needs info from BOTH players AND teams)
        
        Only return: "players", "teams", or "both"."""
        
        response = self.gemini.chat(
            system_prompt="You are an expert database assistant.",
            user_prompt=prompt
        ).lower()
        
        if "both" in response:
            return "both"
        if "team" in response or "club" in response:
            return "teams"
        if "player" in response or "footballer" in response:
            return "players"
        return "players"

    def decompose_query(self, user_question: str) -> dict[str, str]:
        prompt = f"""Given this question: "{user_question}"
        
        This question requires information from BOTH the players table and the teams table.
        Please decompose it into TWO focused sub-questions:
        
        1. A sub-question focused ONLY on player information (name, stats, bio, etc.)
        2. A sub-question focused ONLY on team information (stadium, history, league, etc.)
        
        Return ONLY valid JSON in this exact format:
        {{
            "players": "sub-question for players",
            "teams": "sub-question for teams"
        }}
        
        Example:
        Question: "Which team does Messi play for and where is their stadium?"
        {{
            "players": "Which team does Messi play for?",
            "teams": "Where is Inter Miami's stadium?"
        }}"""
        
        response = self.gemini.chat(
            system_prompt="You are an expert at decomposing complex queries. Return only JSON.",
            user_prompt=prompt
        )
        
        try:
            import json
            # Clean response if needed (sometimes models add markdown code blocks)
            cleaned_response = response.replace("```json", "").replace("```", "").strip()
            decomposed = json.loads(cleaned_response)
            return decomposed
        except:
            # Fallback
            return {
                "players": user_question,
                "teams": user_question
            }

    def retrieve_by_filters(self, query: str, filters: dict | None = None, top_k: int = 5):
        table = self.llm_select_table(query)
        
        if table == "both":
            k = max(1, top_k // 2)
            results_teams = self.supabase.search_by_filters("teams", filters or {}, k)
            results_players = self.supabase.search_by_filters("players", filters or {}, k)
            return results_teams + results_players
            
        return self.supabase.search_by_filters(
            table=table,
            filters=filters or {},
            top_k=top_k,
        )

    def retrieve_semantic(self, query: str, query_embedding: list[float], top_k: int = 5):
        table = self.llm_select_table(query)
        
        if table == "both":
            subqueries = self.decompose_query(query)
            k = max(1, top_k // 2)
            
            # Generate new embeddings for sub-queries
            players_embedding = self.embedding_client.get_embedding(subqueries["players"])
            results_players = self.supabase.search_vectors("players", players_embedding, None, k)
            
            teams_embedding = self.embedding_client.get_embedding(subqueries["teams"])
            results_teams = self.supabase.search_vectors("teams", teams_embedding, None, k)
            
            return results_players + results_teams

        return self.supabase.search_vectors(
            table=table,
            query_embedding=query_embedding,
            filters=None,
            top_k=top_k,
        )

    def retrieve_hybrid(self, query: str, query_embedding: list[float], filters: dict | None = None, top_k: int = 5):
        table = self.llm_select_table(query)
        
        if table == "both":
            subqueries = self.decompose_query(query)
            k = max(1, top_k // 2)
            
            players_embedding = self.embedding_client.get_embedding(subqueries["players"])
            results_players = self.supabase.search_vectors("players", players_embedding, filters, k)
            
            teams_embedding = self.embedding_client.get_embedding(subqueries["teams"])
            results_teams = self.supabase.search_vectors("teams", teams_embedding, filters, k)
            
            return results_players + results_teams

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
        return self.retrieve_hybrid(query, query_embedding, filters, top_k)