from qdrant_client import QdrantClient, models
from fastembed import SparseTextEmbedding
from typing import Optional, List


class Retriever:
    def __init__(
        self,
        client: QdrantClient,
        embedder: SparseTextEmbedding,
        collection_name: str,
        search_limit: Optional[int] = None,
        payload_key: Optional[str] = None,
    ) -> None:
        self.client = client
        self.embedder = embedder
        self.collection_name = collection_name
        self.search_limit = search_limit or 5
        self.payload_key = payload_key or "value"

    def _embed(self, text: str) -> models.NamedSparseVector:
        embeddings = list(self.embedder.embed(text))
        query_vector = models.NamedSparseVector(
            name="text",
            vector=models.SparseVector(
                indices=embeddings[0].indices,
                values=embeddings[0].values,
            ),
        )
        return query_vector

    def _extract_payload(
        self, results: List[models.ScoredPoint], payload_key: Optional[str] = None
    ) -> List[str]:
        pk = self.payload_key or payload_key
        return [result.payload[pk] for result in results]

    def _search(
        self, embeddings: models.NamedSparseVector, limit: Optional[int] = None
    ) -> List[models.ScoredPoint]:
        return self.client.search(
            collection_name=self.collection_name,
            query_vector=embeddings,
            query_filter=None,
            limit=limit or self.search_limit,
        )

    def retrieve(
        self, text: str, limit: Optional[int] = None, payload_key: Optional[str] = None
    ):
        return self._extract_payload(
            self._search(self._embed(text), limit=limit), payload_key=payload_key
        )
