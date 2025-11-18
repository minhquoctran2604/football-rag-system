from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional


class Strategy(str, Enum):
    FILTERS_ONLY = "filters_only"
    SEMANTIC = "semantic"
    HYBRID = "hybrid"


@dataclass
class QueryContext:
    raw_query: str
    strategy: Strategy              
    filters: Dict[str, str]
    embedding: Optional[List[float]]