from .retriever import Retriever
from .generator import ResponseGenerator
from .query_processor import QueryProcessor

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
        qp = self.query_processor(query)
        docs = self.retriever(query_embedding=qp["embedding"], filters=qp["filters"], query=query)
        answer = self.generator(query, docs)
        return {"answer": answer, "context": docs}