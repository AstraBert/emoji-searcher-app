import pandas as pd

from qdrant_client import QdrantClient, models

from fastembed import SparseTextEmbedding

qdrant_client = QdrantClient("http://localhost:6333")
model = SparseTextEmbedding(model_name="Qdrant/minicoil-v1")

df = pd.read_csv("data/emoji.csv")
emojis_texts = df["Description"].to_list()
emojis_image = df["Emoji"].to_list()

embeddings = list(model.embed(emojis_texts))
qdrant_client.create_collection(
    collection_name="emojis_text",
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
            payload={"emoji": emojis_image[idx]},
        )
    )
qdrant_client.upload_points(collection_name="emojis_text", points=points)
