from src.emoji_searcher_app.utils import Retriever
from qdrant_client import QdrantClient, models
from fastembed import SparseTextEmbedding


documents = ["Hello world", "This is a test"]
model = SparseTextEmbedding("Qdrant/minicoil-v1")

qdrant_client = QdrantClient(":memory:")
embeddings = list(model.embed(documents))
qdrant_client.create_collection(
    collection_name="texts",
    vectors_config={},
    sparse_vectors_config={
        "text": models.SparseVectorParams(
            modifier=models.Modifier.IDF,
        ),
    },
)

points = []
for idx, emb in enumerate(embeddings):
    points.append(
        models.PointStruct(
            id=idx,
            vector={
                "text": models.SparseVector(indices=emb.indices, values=emb.values)
            },
            payload={"document": documents[idx]},
        )
    )
qdrant_client.upload_points(collection_name="texts", points=points)


def test_retriever() -> None:
    global qdrant_client, model
    retriever = Retriever(
        client=qdrant_client,
        embedder=model,
        collection_name="texts",
        payload_key="document",
        search_limit=1,
    )
    assert retriever.retrieve("Hello world") == ["Hello world"]
    assert retriever.retrieve("This is a test") == ["This is a test"]
