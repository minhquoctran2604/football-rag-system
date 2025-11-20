from sentence_transformers import SentenceTransformer

class LocalEmbeddingClient:
    def __init__(self, model_name: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"):
        print(f"Loading local embedding model: {model_name}...")
        self.model = SentenceTransformer(model_name)
        print("Model loaded successfully.")

    def get_embedding(self, text: str) -> list[float]:
        if not text:
            return []
        embedding = self.model.encode(text)
        return embedding.tolist()
