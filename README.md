# Grounded Document Q&A Bot

A beginner-friendly Retrieval-Augmented Generation (RAG) bot that answers questions from a local document collection and shows the source chunks it used. It ingests PDFs and text files, chunks them with overlap, embeds them in batches, stores vectors in a persistent FAISS index, then uses either Ollama or OpenAI to produce grounded answers with citations.

## Tech Stack

- Python 3.11+
- A built-in hashing TF-IDF style embedding model for local retrieval
- `faiss-cpu==1.8.0.post1` for the persistent vector database
- `pypdf==4.2.0` for PDF text extraction
- `python-docx==1.1.2` for optional DOCX ingestion
- `openai==1.35.13` for optional OpenAI answer generation
- `requests==2.32.3` for Ollama HTTP calls
- `python-dotenv==1.0.1` for environment variables
- `numpy==1.26.4` for vector handling

## Architecture Overview

```text
data/ documents
   -> document loader extracts clean text and page metadata
   -> paragraph-aware chunker creates overlapping chunks
   -> local hashing embedder embeds chunks in batches
   -> FAISS stores vectors on disk with JSON metadata
   -> query command embeds the question and retrieves top-k chunks
   -> LLM receives only retrieved context and returns a cited answer
```

The indexing and querying steps are intentionally separate. Indexing writes `storage/faiss.index` and `storage/chunks.json`, while querying only loads those persisted files.

## Chunking Strategy

This project uses paragraph-aware fixed-size chunking. Paragraphs are accumulated until they reach about 900 characters, with a 180-character overlap between neighboring chunks. This keeps related sentences together better than a pure character splitter while still giving predictable chunk sizes for retrieval. The overlap helps preserve context when useful details fall near a chunk boundary.

Every chunk stores:

- Source filename
- Page number when available
- Chunk number
- Original chunk text

## Embedding Model and Vector Database

The embedding model is a built-in hashing TF-IDF style embedder. It tokenizes text, maps tokens into a fixed 768-dimensional vector with a stable hash, applies light inverse-frequency weighting learned from the indexed chunks, and normalizes vectors for cosine similarity. This keeps the assignment fully runnable without downloading a separate embedding model, while still demonstrating batched embedding and vector search clearly.

FAISS is used as the vector database because it is lightweight, fast, and easy to persist for an internship assignment. The FAISS index is saved to disk so the documents are not re-indexed every time the bot starts.

## Setup

1. Clone the repository:

```bash
git clone <your-public-repo-url>
cd <repo-folder>
```

2. Create and activate a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
```

On macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Copy the example environment file:

```bash
copy .env.example .env
```

On macOS or Linux:

```bash
cp .env.example .env
```

5. Choose an LLM provider.

For Ollama, install Ollama and pull a model:

```bash
ollama pull llama3.1
```

Then set this in `.env`:

```text
LLM_PROVIDER=ollama
OLLAMA_MODEL=llama3.1
```

For OpenAI, set:

```text
LLM_PROVIDER=openai
OPENAI_API_KEY=your_key_here
OPENAI_MODEL=gpt-4o-mini
```

6. Build the vector index:

```bash
python -m rag_bot.index --data-dir data --storage-dir storage
```

7. Start the command-line Q&A bot:

```bash
python -m rag_bot.chat --storage-dir storage --top-k 4
```

Type `exit` or `quit` to stop the bot.

8. Optional: run the browser UI:

```bash
python -m rag_bot.web --data-dir data --storage-dir storage --port 8000
```

Then open:

```text
http://127.0.0.1:8000
```

The web UI includes document upload/paste, automatic re-indexing, a question box, configurable top-k retrieval, sample questions, generated answers, and the retrieved source chunks with citations.

## Environment Variables

Never commit real API keys. Use `.env` locally.

| Variable | Required | Description |
| --- | --- | --- |
| `LLM_PROVIDER` | Yes | `ollama` or `openai` |
| `OLLAMA_BASE_URL` | No | Defaults to `http://localhost:11434` |
| `OLLAMA_MODEL` | If using Ollama | Defaults to `llama3.1` |
| `OPENAI_API_KEY` | If using OpenAI | Your OpenAI API key |
| `OPENAI_MODEL` | If using OpenAI | Defaults to `gpt-4o-mini` |

## Example Queries

- How can urban tree canopy reduce heat risk in cities?
- What are the main reliability challenges for renewable microgrids?
- How is machine learning used in financial fraud detection?
- What privacy principles should a small business follow when collecting customer data?
- What makes AI useful but risky in healthcare settings?
- What is the author of the documents' favorite movie?

The final question is intentionally not answerable from the documents. The bot should say that the retrieved context does not contain enough information.

## Known Limitations

- The local hashing embedder is lexical rather than deeply semantic, so it works best when questions share important terms with the documents.
- The sample corpus is small, so answers are limited to the included documents.
- PDF extraction quality depends on how the PDF stores text.
- FAISS similarity search does not understand citations by itself; citations come from metadata attached during ingestion.
- The bot relies on the chosen LLM to synthesize the final answer, so the prompt explicitly tells it not to use outside knowledge.
- The web UI is intentionally simple and uses Python's built-in HTTP server rather than a larger frontend framework.

## Suggested Screen Recording Flow

1. Show the folder structure: `data/`, `rag_bot/`, `storage/`, and configuration files.
2. Run `python -m rag_bot.index --data-dir data --storage-dir storage`.
3. Run `python -m rag_bot.chat --storage-dir storage --top-k 4`.
4. Ask at least five questions from the README examples.
5. Ask the unanswerable favorite-movie question and show the refusal.
6. Briefly explain the paragraph-aware chunking and why FAISS was chosen.
