from __future__ import annotations
from typing import Any, Dict, List, Optional
from src.utils.gemini_client import GeminiClient
from .types import Strategy

class ResponseGenerator:
    def __init__(self, gemini_client: GeminiClient) -> None:
        self.gemini = gemini_client

    def _format_doc(self, idx: int, doc: Dict[str, Any]) -> str:

        fields_to_skip = {"embedding"}
        clean_doc = {k: v for k, v in doc.items() if k not in fields_to_skip}

        lines = [f"[Doc {idx}]"]
        for k, v in clean_doc.items():
            lines.append(f"{k}: {v}")
        return "\n".join(lines)

    def _build_context(self, docs: List[Dict[str, Any]]) -> str:
        
        if not docs:
            return "No documents were retrieved from the database."

        formatted = [self._format_doc(i + 1, doc) for i, doc in enumerate(docs)] # gộp các doc đã format
        return "\n\n".join(formatted)


    def __call__(self, query: str, docs: List[Dict[str, Any]],
                strategy: Optional[Strategy] = None,
                filters: Optional[Dict[str, Any]] = None) -> str:
        context_block = self._build_context(docs)

        system_prompt = (
            "You are a football data assistant. "
            "You answer questions about football players and teams "
            "using ONLY the provided context. "
            "If the answer is not in the context or is unclear, say you don't know. "
            "Always answer in the same language as the user's question."
        )

        user_prompt = f"""
                        User question:
                        {query}

                        Context from database:
                        {context_block}

                        Instructions:
                        - Use only the information in the context above.
                        - If you are not sure, explicitly say you are not sure instead of guessing.
                        - Provide a concise but complete answer.
                        """
        response = self.gemini.chat(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )

        return response
