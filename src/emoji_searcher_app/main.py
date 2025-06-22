from qdrant_client import QdrantClient
from fastembed import SparseTextEmbedding
from utils import Retriever
from typing import List
import gradio as gr

QC = QdrantClient("http://localhost:6333")
COLLECTION_NAME = "emojis_text"
MODEL = SparseTextEmbedding(model_name="Qdrant/minicoil-v1")

RETRIEVER = Retriever(
    client=QC, collection_name=COLLECTION_NAME, embedder=MODEL, payload_key="emoji"
)


def search_emoji(text: str) -> List[str]:
    res = RETRIEVER.retrieve(text)
    if len(res) < 5:
        for j in range(len(res), 5):
            res.append("**No result**")
    return res


iface = gr.Interface(
    fn=search_emoji,
    inputs=[gr.Textbox(label="Emoji Name")],
    outputs=[gr.Markdown(container=True, show_copy_button=True) for i in range(5)],
    title="Emoji Search",
    theme=gr.themes.Citrus(),
    submit_btn="Search",
    clear_btn=None,
)

if __name__ == "__main__":
    iface.launch(pwa=True)
