# Classic RAG PDF Chat

A compact, step-by-step Retrieval-Augmented Generation project. It extracts text from uploaded PDFs, splits it into overlapping chunks, embeds each chunk locally, retrieves the four closest chunks with FAISS or ChromaDB, then asks an OpenAI-compatible chat model to answer from that context. Gradio provides the upload and chat interface.

## How the pipeline works

```text
PDFs → extract text by page → split into overlapping chunks → local embeddings → FAISS / ChromaDB
Question → local embedding → top 4 matching chunks → prompt + question → LLM → cited answer
```

## 1. Create an environment and install dependencies

Python 3.10–3.12 is recommended. From this folder:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

`sentence-transformers` downloads its embedding model on first use. FAISS wheels can vary by operating system and Python version; if `faiss-cpu` is unavailable for your setup, select ChromaDB in the app and install with `pip install gradio pypdf sentence-transformers chromadb openai`.

## 2. Configure an LLM

For OpenAI, either set `OPENAI_API_KEY` in your shell or enter it in the app's **LLM settings**. The app defaults to `gpt-4o-mini`. The key is not saved to disk.

For local Ollama, install and start Ollama, pull a chat model (for example `ollama pull llama3.2`), choose **Ollama**, and keep the model name in the settings. Ollama's OpenAI-compatible endpoint is expected at `http://localhost:11434/v1`.

## 3. Run the app

```powershell
python app.py
```

Open the local Gradio URL printed in the terminal. Select a vector store, upload one or more text-based PDFs, and click **Build / replace index**. Once the status says indexing is complete, ask a question in the chat.

## 4. Follow the code path

1. `extract_pdf()` reads each page with `pypdf` and keeps the page number as metadata.
2. `split_text()` makes 1,000-character chunks with 150-character overlap.
3. `get_embedder()` loads `all-MiniLM-L6-v2` locally and embeds all chunks.
4. `build_index()` writes normalized vectors to FAISS or a persistent Chroma collection under `indexes/`.
5. `retrieve()` embeds a question and returns its four nearest chunks.
6. `answer()` sends those chunks and the question to the selected LLM and appends source/page references.

## Notes

- Scanned/image-only PDFs need OCR before this app can retrieve their text.
- The default embedding model runs locally; the selected chat provider receives the question and retrieved passages.
- This is a teaching-sized single-user example. It does not include authentication, document-level access controls, OCR, or production monitoring.
- To start fresh, use **Build / replace index** with the PDFs you want indexed. Chroma replaces its collection; FAISS replaces its saved index.