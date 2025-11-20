from __future__ import annotations

import json
from typing import Dict, Any

from src.utils.gemini_client import GeminiClient
from src.rag.types import QueryContext, Strategy


class QueryProcessor:
    """Tạo embedding có điều kiện và trích xuất filters + strategy."""

    def __init__(self, gemini_client: GeminiClient, embedding_client: Any):
        self.gemini = gemini_client
        self.embedding_client = embedding_client

        # Prompt trích filters (copy lại prompt cũ đầy đủ của bạn)
        self.system_prompt_filters = """
You are a highly specialized entity extractor for a football database.
Your one and only task is to identify specific entities from the user's query.

The only entities you are allowed to identify are 'league' and 'nationality'.

You MUST return a valid JSON object.
- If you find one or more supported entities, return them as key-value pairs.
- If no supported entities are found in the query, you MUST return an empty JSON object: {}.
- Do NOT invent any other keys. Only use 'league' and 'nationality'.

Example 1:
Query: "Find Argentinian players in La Liga"
Response: {"league": "La Liga", "nationality": "Argentina"}

Example 2:
Query: "Who is the best player in the world?"
Response: {}

Example 3:
Query: "Show me goalkeepers from Germany"
Response: {"nationality": "Germany"}
"""

        # Prompt phân loại strategy
        self.system_prompt_strategy = """
Phân loại query football theo 3 loại:
- filters_only: chỉ chứa thông tin league/nationality (không mô tả skill)
- semantic: chỉ mô tả skill/đặc điểm, không có league/nationality
- hybrid: vừa có league/nationality, vừa có mô tả skill

Examples:
"Italian players in La Liga" → filters_only
"Show me players from Germany" → filters_only
"fast winger good at dribbling" → semantic
"playmaker with great passing vision" → semantic
"Italian striker with good finishing" → hybrid
"Brazilian winger in Premier League, very fast" → hybrid

Chỉ trả về đúng một trong 3 từ sau:
filters_only
semantic
hybrid

Query: "{query}"
Filters: {filters}
Response:
"""

    def _extract_filters(self, query: str) -> Dict[str, str]:
        filters: Dict[str, str] = {}
        try:
            response_text = self.gemini.chat(
                system_prompt=self.system_prompt_filters,
                user_prompt=f'Query: "{query}"\nResponse:'
            )
            clean_response = (
                response_text.strip()
                .replace("```json", "")
                .replace("```", "")
                .strip()
            )
            parsed_filters = json.loads(clean_response)
            allowed_keys = {"league", "nationality"}
            filters = {
                key: value
                for key, value in parsed_filters.items()
                if key in allowed_keys
            }
        except (json.JSONDecodeError, TypeError):
            print("Warning: Could not parse filters from LLM response.")
            filters = {}
        return filters

    def _decide_strategy(self, query: str, filters: Dict[str, str]) -> Strategy:
        user_prompt = self.system_prompt_strategy.format(query=query, filters=filters)
        response = self.gemini.chat(
            system_prompt="Expert classifier.",
            user_prompt=user_prompt
        ).strip().lower()

        token = response.split()[0]

        if "filters_only" in token:
            return Strategy.FILTERS_ONLY
        elif "semantic" in token:
            return Strategy.SEMANTIC
        else:
            # Mặc định: hybrid
            return Strategy.HYBRID

    def __call__(self, query: str) -> QueryContext:
        """Main call: extract → decide → conditional embedding → return QueryContext."""
        filters = self._extract_filters(query)
        strategy = self._decide_strategy(query, filters)

        embedding = (
            self.embedding_client.get_embedding(query)
            if strategy in (Strategy.SEMANTIC, Strategy.HYBRID)
            else None
        )

        return QueryContext(
            raw_query=query,
            filters=filters,
            strategy=strategy,
            embedding=embedding,
        )