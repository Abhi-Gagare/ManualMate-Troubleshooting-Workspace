# ManualMate

**A manual-powered troubleshooting workspace.** Upload equipment manuals, describe a problem, and find relevant instructions with references to the source pages.

ManualMate is built for situations where a manual is available but the right troubleshooting section is hard to find. It can be used with printer, appliance, computer hardware, and other equipment manuals. Answers depend on the contents and quality of the uploaded documents; check the cited instructions before acting.

## How it works

1. Upload one or more PDF manuals, or load a saved manual library.
2. Select the manuals to search and describe the symptoms, device model, or error code.
3. ManualMate searches the selected manuals using semantic and keyword retrieval.
4. The chosen answer method returns relevant excerpts or a generated response, with page references and a source viewer.

The **Home** page introduces the project and links to the workspace. Use **Troubleshoot** to ask questions, **Manual Library** to upload and manage saved libraries, and **Settings** to choose an answer method and search options. The current library and conversation stay available as you move between pages.

When a language model is selected, it is instructed to use only the retrieved manual passages and organize supported troubleshooting guidance into possible causes, steps, and safety instructions. If no passages pass the relevance threshold, the app says it could not find the information. Simple Extraction instead displays matching excerpts directly. Generated answers and citations are not independently verified, so consult the manual before performing a procedure.

## Features

- Multi-PDF manual library with content deduplication and optional saved indexes.
- Page-aware extraction, section metadata, ruled-table extraction, and optional OCR for scanned pages.
- Multilingual local embeddings and hybrid FAISS/BM25 search, with optional reranking.
- Manual selection, adjustable relevance threshold, and source passages with page references.
- Local extractive answers, local Ollama models, or Groq, Hugging Face, and Together AI providers.
- Conversation export to Markdown or PDF. PDF export uses a built-in Latin-1 font, so other scripts may be replaced; use Markdown for Arabic or other unsupported text.

## Getting started

Requirements: Python 3.10 or newer. Tesseract with the `eng` and `ara` language data is optional for scanned manuals.

```bash
python -m venv .venv
# Windows
.venv\\Scripts\\activate
# macOS / Linux
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Open the local URL printed by Streamlit. The first use of the default embedding model, `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, downloads the model. Later runs use the cached files. To use another model, set `EMBEDDING_MODEL` before launching. Saved libraries made with a different model are re-embedded when opened.

### Answer methods

- **Simple Extraction:** no API key or language model server; returns matching manual excerpts.
- **Ollama:** use a locally running Ollama server and an installed model, such as `ollama pull llama3.2`.
- **Cloud providers:** supply the relevant API key in the sidebar or through `GROQ_API_KEY`, `HUGGINGFACE_API_KEY`, or `TOGETHER_API_KEY`. Retrieved passages are sent to the selected provider.

See `.env.example` for model and retrieval settings. To load a `.env` file automatically, install the optional `python-dotenv` package; otherwise set environment variables in your shell.

## Search and storage

PyMuPDF extracts PDF content, using Tesseract OCR on pages that appear to be scanned when available. Text is chunked without crossing pages; chunks are embedded locally and added to a FAISS index. BM25 provides keyword matching, and reciprocal rank fusion combines both rankings. An optional cross-encoder reranks candidates.

Uploads are processed in memory. Saving a library explicitly writes its extracted text, metadata, and vector index to `data/<library-name>/` by default. Treat saved libraries as sensitive as the original manuals. The app does not save uploaded PDF files themselves. Model downloads may require network access even when using a local answer method.

## Project structure

- `app.py` — Streamlit workspace, manual library, troubleshooting conversation, and exports.
- `rag.py` — PDF extraction, OCR, chunking, search, and index persistence.
- `providers.py` — answer instructions, provider streaming, and citation formatting.
- `.env.example` — optional configuration.

## Limitations

OCR, reading order, and table detection are best-effort, particularly for complex layouts and borderless tables. The application retrieves passages; it does not validate repairs or guarantee that a generated answer is correct. It also cannot determine whether a procedure is appropriate for a particular device condition. Check the original manual and seek qualified help when needed.

## License

[MIT](LICENSE). Retain the license and any required attribution when redistributing the project or its underlying code.
