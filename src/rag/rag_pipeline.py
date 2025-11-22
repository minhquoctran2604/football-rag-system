from .retriever import Retriever
from .generator import ResponseGenerator
from .query_processor import QueryProcessor
from .types import Strategy  # import Enum Strategy


class RAGPipeline:
    def __init__(
        self,
        retriever: Retriever,
        generator: ResponseGenerator,
        query_processor: QueryProcessor,
    ):
        self.retriever = retriever
        self.generator = generator
        self.query_processor = query_processor

    def __call__(self, query: str) -> dict:
        # query_processor tra ve QueryContext object
        qp = self.query_processor(query)
        strategy: Strategy = qp.strategy
        embedding = qp.embedding
        filters = qp.filters
        sort_field = qp.sort_field
        sort_order = qp.sort_order

        # 1) lay docs theo strategy
        docs = self._retrieve(
            query=query,
            strategy=strategy,
            embedding=embedding,
            filters=filters,
            sort_field=sort_field,
            sort_order=sort_order,
        )

        # 2) generate cau tra loi
        answer = self.generator(
            query=query, 
            docs=docs or [],  # Fix: Fallback to empty list if None
            strategy=strategy, 
            filters=filters
        )

        # tra ve them strategy/filters
        return {
            "answer": answer,
            "context": docs or [],  # Fix: Consistent with generator input
            "strategy": strategy.value,
            "filters": filters or {},
        }

    def _retrieve(
        self,
        query: str,
        strategy: Strategy,
        embedding: list[float] | None,
        filters: dict | None,
        sort_field: str | None,
        sort_order: str | None,
    ) -> list[dict]:

        if strategy == Strategy.FILTERS_ONLY:
            # chi dung filters
            return self.retriever.retrieve_by_filters(
                query=query,
                filters=filters or {},
            )

        if strategy == Strategy.SEMANTIC:
            # chi dung embedding
            if embedding is None:
                raise ValueError("Semantic strategy requires embedding.")
            return self.retriever.retrieve_semantic(
                query=query,
                query_embedding=embedding,
            )

        if strategy == Strategy.RANKING:
            # chi dung filters va sort
            if not sort_field or not sort_order:
                raise ValueError("RANKING strategy requires sort_field and sort_order")
            return self.retriever.retrieve_ranking(
                query=query,
                filters=filters or {},
                sort_field=sort_field,
                sort_order=sort_order,
            )

        # mac dinh: HYBRID
        if embedding is None:
            raise ValueError("Hybrid strategy requires embedding.")
        return self.retriever.retrieve_hybrid(
            query=query,
            query_embedding=embedding,
            filters=filters,
        )
