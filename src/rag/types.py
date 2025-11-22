from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Literal


class Strategy(str, Enum):
    FILTERS_ONLY = "filters_only"
    SEMANTIC = "semantic"
    HYBRID = "hybrid"
    RANKING = 'ranking'

@dataclass
class QueryContext:
    raw_query: str
    strategy: Strategy              
    filters: Dict[str, str]
    embedding: Optional[List[float]]
    sort_field: Optional[str] = None # attr in json col to sort
    sort_order: Optional[Literal["DESC", "ASC"]] = None
