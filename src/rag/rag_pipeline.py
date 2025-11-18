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
        # Use dataclass attributes instead of dictionary access
        strategy: Strategy = qp.strategy
        embedding = qp.embedding
        filters = qp.filters

        # 1) lay docs theo strategy
        docs = self._retrieve(
            query=query,
            strategy=strategy,
            embedding=embedding,
            filters=filters,
        )

        # 2) generate cau tra loi
        answer = self.generator(
            query=query,
            docs=docs,
            strategy=strategy,  
            filters=filters     
        )

        # tra ve them strategy/filters
        return {
            "answer": answer,
            "context": docs,
            "strategy": strategy.value,
            "filters": filters or {},
        }

    def _retrieve(
        self,
        query: str,
        strategy: Strategy,
        embedding: list[float] | None,
        filters: dict | None,
    ):

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

        # mac dinh: HYBRID
        if embedding is None:
            raise ValueError("Hybrid strategy requires embedding.")
        return self.retriever.retrieve_hybrid(
            query=query,
            query_embedding=embedding,
            filters=filters,
        )