"""A small, inspectable Classic RAG app with Gradio, PDF ingestion, and two stores."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import gradio as gr
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()


APP_DIR = Path(__file__).parent
DATA_DIR = APP_DIR / "data"
INDEX_DIR = APP_DIR / "indexes"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
TOP_K = 4

_embedder: SentenceTransformer | None = None
_faiss_index: Any = None
_faiss_chunks: list[dict[str, Any]] = []


def get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBEDDING_MODEL)
    return _embedder


def split_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Simple character-window chunking, with word boundaries where possible."""
    text = " ".join(text.split())
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            boundary = text.rfind(" ", start, end)
            if boundary > start + size // 2:
                end = boundary
        if text[start:end].strip():
            chunks.append(text[start:end].strip())
        if end == len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def extract_pdf(file_path: str) -> list[dict[str, Any]]:
    reader = PdfReader(file_path)
    name = Path(file_path).name
    output: list[dict[str, Any]] = []
    for page_no, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""
        for chunk_no, chunk in enumerate(split_text(page_text)):
            output.append({"text": chunk, "source": name, "page": page_no, "chunk": chunk_no})
    return output


def build_index(files: list[Any] | None, backend: str) -> str:
    global _faiss_index, _faiss_chunks
    if not files:
        return "Upload at least one PDF first."
    paths = [f.name if hasattr(f, "name") else str(f) for f in files]
    chunks = [chunk for path in paths for chunk in extract_pdf(path)]
    if not chunks:
        return "No extractable text found. Scanned PDFs need OCR before indexing."

    # Both stores use the same normalized sentence-transformer vectors.
    vectors = get_embedder().encode([c["text"] for c in chunks], normalize_embeddings=True,
                                    convert_to_numpy=True, show_progress_bar=False)
    INDEX_DIR.mkdir(exist_ok=True)
    if backend == "FAISS":
        import faiss

        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors.astype("float32"))
        faiss.write_index(index, str(INDEX_DIR / "classic_rag.faiss"))
        import json

        (INDEX_DIR / "classic_rag_chunks.json").write_text(json.dumps(chunks, ensure_ascii=False), encoding="utf-8")
        _faiss_index, _faiss_chunks = index, chunks
    else:
        import chromadb

        client = chromadb.PersistentClient(path=str(INDEX_DIR / "chroma"))
        try:
            client.delete_collection("pdf_chunks")
        except Exception:
            pass
        collection = client.create_collection("pdf_chunks", metadata={"hnsw:space": "cosine"})
        collection.add(ids=[f"chunk-{i}" for i in range(len(chunks))],
                       embeddings=vectors.tolist(), documents=[c["text"] for c in chunks],
                       metadatas=[{k: v for k, v in c.items() if k != "text"} for c in chunks])
    return f"Indexed {len(chunks)} chunks from {len(paths)} PDF(s) in {backend}."


def retrieve(question: str, backend: str) -> list[dict[str, Any]]:
    query = get_embedder().encode([question], normalize_embeddings=True, convert_to_numpy=True).astype("float32")
    if backend == "FAISS":
        global _faiss_index, _faiss_chunks
        index_file = INDEX_DIR / "classic_rag.faiss"
        chunks_file = INDEX_DIR / "classic_rag_chunks.json"
        if _faiss_index is None and index_file.exists() and chunks_file.exists():
            import faiss
            import json

            _faiss_index = faiss.read_index(str(index_file))
            _faiss_chunks = json.loads(chunks_file.read_text(encoding="utf-8"))
        if _faiss_index is None:
            return []
        scores, ids = _faiss_index.search(query, min(TOP_K, len(_faiss_chunks)))
        return [{**_faiss_chunks[i], "score": float(score)} for score, i in zip(scores[0], ids[0]) if i >= 0]

    import chromadb

    client = chromadb.PersistentClient(path=str(INDEX_DIR / "chroma"))
    try:
        collection = client.get_collection("pdf_chunks")
    except Exception:
        return []
    result = collection.query(query_embeddings=query.tolist(), n_results=TOP_K, include=["documents", "metadatas", "distances"])
    found = []
    for doc, meta, distance in zip(result["documents"][0], result["metadatas"][0], result["distances"][0]):
        found.append({"text": doc, **meta, "score": 1.0 - float(distance)})
    return found


def answer(question: str, history: list[dict[str, str]] | None, backend: str,
           provider: str, model: str, api_key: str) -> str:
    if not question.strip():
        return "Please enter a question."
    sources = retrieve(question, backend)
    if not sources:
        return "No index found. Upload PDFs and click **Build / replace index** first."
    context = "\n\n".join(
        f"[Source: {s['source']}, page {s['page']}]\n{s['text']}" for s in sources
    )
    prompt = (
        "Answer using only the supplied context. If it does not contain the answer, "
        "say you could not find it in the uploaded PDFs. Cite claims with the source "
        "filename and page number.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}"
    )
    if provider == "Gemini":
        key = api_key or os.getenv("GEMINI_API_KEY", "")
        if not key:
            return "Add your Gemini API key in LLM settings or set GEMINI_API_KEY in .env, then ask again."
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=key)
        response = client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.2),
        )
        text = response.text or "I could not produce an answer."
    else:
        key = api_key or "ollama"
        from openai import OpenAI

        client = OpenAI(api_key=key, base_url="http://localhost:11434/v1")
        completion = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        text = completion.choices[0].message.content or "I could not produce an answer."
    cited_sources = "\n\n**Retrieved passages**\n" + "\n".join(
        f"- {s['source']}, p. {s['page']} (similarity {s['score']:.2f})" for s in sources
    )
    return text + cited_sources


def model_default(provider: str) -> str:
    return "Qwen2.5:3b" if provider == "Ollama" else "gemini-3.8-flash"


with gr.Blocks(title="Classic RAG · PDF Q&A", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# Classic RAG: ask questions about your PDFs\nUpload PDFs, build a vector index, then chat with retrieved passages as context.")
    with gr.Row():
        backend = gr.Dropdown(["ChromaDB", "FAISS"], value="ChromaDB", label="Vector store",
                              info="ChromaDB installs from requirements.txt. Install faiss-cpu separately to use FAISS.")
        files = gr.File(file_count="multiple", file_types=[".pdf"], label="PDF files", type="filepath")
    build = gr.Button("Build / replace index", variant="primary")
    status = gr.Markdown("Upload files and build an index to begin.")
    build.click(build_index, inputs=[files, backend], outputs=status)
    with gr.Accordion("LLM settings", open=False):
        provider = gr.Radio(["Gemini", "Ollama"], value="Gemini", label="Provider")
        model = gr.Textbox(value="gemini-3.8-flash", label="Model", info="Gemini model name or an installed Ollama model")
        api_key = gr.Textbox(value="", type="password", label="Gemini API key", placeholder="Optional if GEMINI_API_KEY is set in .env")
        provider.change(model_default, inputs=provider, outputs=model)
    chatbot = gr.ChatInterface(
        fn=answer,
        additional_inputs=[backend, provider, model, api_key],
        title="Chat with your documents",
        description="Answers are grounded in the top four retrieved PDF chunks and include page citations.",
    )


if __name__ == "__main__":
    demo.launch()
