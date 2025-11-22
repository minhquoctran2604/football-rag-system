from __future__ import annotations
import json
from typing import Dict, Any
from src.utils.gemini_client import GeminiClient
from src.rag.types import QueryContext, Strategy


class QueryProcessor:
    # phân tích query chọn chiến lược tối ưu 

    def __init__(self, gemini_client: GeminiClient, embedding_client: Any):
        self.gemini = gemini_client
        self.embedding_client = embedding_client
        self.router_prompt = """
You are a Query Router for a football RAG system.
Analyze the user's query and decide the best retrieval strategy.

**Strategies:**
1. `ranking`: When user asks for rankings, comparisons, superlatives ("most", "least", "top", "best", "highest", "lowest", "nhiều nhất", "ít nhất", "trẻ nhất", "cao nhất").
2. `filters_only`: When user asks for a list based on explicit attributes only ("Players from Brazil in La Liga").
3. `semantic`: When user describes playing style, skills, or vague concepts ("Fast winger with good dribbling").
4. `hybrid`: When user combines explicit filters with semantic description ("Brazilian striker who is good at headers").

**Output Format (JSON Only):**
{
  "strategy": "ranking" | "filters_only" | "semantic" | "hybrid",
  "filters": {
    "league": "League Name" | null,
    "nationality": "Country Name" | null
  },
  "sort": {
    "field": "goals" | "assists" | "age" | "height" | "appearances" | null,
    "order": "DESC" | "ASC"
  }
}

**Rules:**
- For `ranking` strategy: `sort.field` and `sort.order` are REQUIRED.
- `sort.order` = "DESC" for "most/highest/nhiều nhất", "ASC" for "least/youngest/ít nhất/trẻ nhất".
- `filters` should only extract 'league' and 'nationality'.
- Return ONLY valid JSON.

**Examples:**
Query: "Ai ghi nhiều bàn nhất EPL?"
Response: {"strategy": "ranking", "filters": {"league": "Premier League", "nationality": null}, "sort": {"field": "goals", "order": "DESC"}}

Query: "Cầu thủ trẻ nhất?"
Response: {"strategy": "ranking", "filters": {"league": null, "nationality": null}, "sort": {"field": "age", "order": "ASC"}}

Query: "Tiền đạo Brazil ở La Liga"
Response: {"strategy": "filters_only", "filters": {"league": "La Liga", "nationality": "Brazil"}, "sort": {"field": null, "order": null}}

Query: "Cầu thủ chạy nhanh và sút tốt"
Response: {"strategy": "semantic", "filters": {"league": null, "nationality": null}, "sort": {"field": null, "order": null}}
"""

    def _analyze_query(self, query: str) -> Dict[str, Any]:
        try:
            response_text = self.gemini.chat(
                system_prompt=self.router_prompt,
                user_prompt=f'Query: "{query}"\nJSON Response:'
            )
            # Clean markdown code blocks
            clean_response = (
                response_text.strip()
                .replace("```json", "")
                .replace("```", "")
                .strip()
            )
            return json.loads(clean_response)
        except (json.JSONDecodeError, TypeError) as e:
            print(f"⚠️ Router Error: {e}. Defaulting to hybrid strategy.")
            return {
                "strategy": "hybrid",
                "filters": {"league": None, "nationality": None},
                "sort": {"field": None, "order": None}
            }

    def __call__(self, query: str) -> QueryContext:
        analysis = self._analyze_query(query)

        strategy_str = analysis.get("strategy", "hybrid")
        try:
            strategy = Strategy(strategy_str)
        except ValueError:
            print(f"⚠️ Unknown strategy '{strategy_str}', defaulting to HYBRID")
            strategy = Strategy.HYBRID

        filters_raw = analysis.get("filters") or {}
        filters = {k: v for k, v in filters_raw.items() if v}  # Remove nulls

        sort_info = analysis.get("sort") or {}
        sort_field = sort_info.get("field")  # "goals", "age", etc.
        sort_order = sort_info.get("order")  # "DESC" or "ASC"

        embedding = None
        if strategy in (Strategy.SEMANTIC, Strategy.HYBRID):
            embedding = self.embedding_client.get_embedding(query)

        return QueryContext(
            raw_query=query,
            strategy=strategy,
            filters=filters,
            embedding=embedding,
            sort_field=sort_field,
            sort_order=sort_order
        )