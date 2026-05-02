from __future__ import annotations

import argparse
import base64
import json
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from rag_bot.chunker import chunk_documents
from rag_bot.config import load_settings
from rag_bot.document_loader import SUPPORTED_EXTENSIONS, load_documents
from rag_bot.embeddings import EmbeddingModel
from rag_bot.llm import LlmClient
from rag_bot.prompts import build_user_prompt
from rag_bot.vector_store import FaissVectorStore


HTML_PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>RAG Document Q&A Bot</title>
  <style>
    :root {
      color-scheme: light;
      font-family: "Segoe UI", Inter, Arial, sans-serif;
      color: #18202f;
      --accent: #2563eb;
      --accent-strong: #1d4ed8;
      --teal: #0f766e;
      --amber: #f59e0b;
      --ink: #18202f;
      --muted: #667085;
      --line: #d8dee8;
      --soft-line: #e7ebf1;
      --panel: #ffffff;
      --subtle: #f8fafc;
      --warning: #b45309;
      --danger: #b42318;
      --shadow: 0 16px 38px rgba(30, 41, 59, 0.10);
    }
    body {
      margin: 0;
      min-height: 100vh;
      background: #eef3f8;
    }
    main {
      max-width: 1220px;
      margin: 0 auto;
      padding: 24px;
    }
    header {
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 18px;
      align-items: center;
      margin-bottom: 18px;
      padding: 22px;
      border-radius: 18px;
      color: white;
      background: linear-gradient(135deg, #172554, #2563eb 48%, #0f766e);
      box-shadow: 0 22px 50px rgba(37, 99, 235, 0.24);
    }
    .brand-kicker {
      color: rgba(255, 255, 255, 0.78);
      font-size: 13px;
      font-weight: 750;
      margin-bottom: 8px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }
    h1 {
      margin: 0;
      font-size: 34px;
      line-height: 1.1;
      letter-spacing: 0;
      color: #ffffff;
    }
    p {
      margin: 0;
      color: var(--muted);
      line-height: 1.5;
    }
    header p {
      margin-top: 8px;
      color: rgba(255, 255, 255, 0.84);
      max-width: 620px;
    }
    .status {
      display: flex;
      gap: 7px;
      flex-wrap: wrap;
      justify-content: flex-end;
      font-size: 13px;
      color: var(--muted);
      position: relative;
      z-index: 1;
    }
    .status span {
      border: 1px solid rgba(255, 255, 255, 0.32);
      background: rgba(255, 255, 255, 0.14);
      border-radius: 7px;
      padding: 7px 9px;
      white-space: nowrap;
      color: rgba(255, 255, 255, 0.86);
      backdrop-filter: blur(8px);
    }
    .status strong {
      color: #ffffff;
    }
    .layout {
      display: grid;
      grid-template-columns: 360px minmax(0, 1fr);
      gap: 18px;
      align-items: start;
    }
    .workspace {
      display: grid;
      gap: 16px;
    }
    form {
      display: grid;
      gap: 10px;
    }
    .field-label {
      display: block;
      margin-bottom: 6px;
      font-size: 13px;
      font-weight: 750;
      color: #303b4f;
    }
    textarea,
    .file-row input[type="text"] {
      width: 100%;
      box-sizing: border-box;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
      font: inherit;
      background: white;
      color: var(--ink);
      outline: none;
      transition: border-color 0.15s ease, box-shadow 0.15s ease;
    }
    textarea:focus,
    .file-row input[type="text"]:focus,
    input:focus {
      border-color: var(--accent);
      box-shadow: 0 0 0 3px rgba(15, 118, 110, 0.13);
    }
    textarea {
      min-height: 118px;
      resize: vertical;
    }
    #document-text {
      min-height: 92px;
    }
    .file-row {
      display: grid;
      gap: 8px;
    }
    .file-row input[type="file"] {
      width: 100%;
      box-sizing: border-box;
      border: 1px dashed #aeb8c7;
      border-radius: 12px;
      padding: 14px;
      background: var(--subtle);
      color: #3f4a5f;
    }
    .controls {
      display: flex;
      gap: 10px;
      align-items: center;
      flex-wrap: wrap;
      justify-content: space-between;
    }
    button {
      border: 0;
      border-radius: 8px;
      background: linear-gradient(135deg, var(--accent), var(--teal));
      color: white;
      font-weight: 700;
      padding: 11px 18px;
      cursor: pointer;
      transition: background 0.15s ease, transform 0.15s ease;
    }
    button:hover {
      background: linear-gradient(135deg, var(--accent-strong), #0b5f59);
      transform: translateY(-1px);
    }
    button:disabled {
      background: #8bb8b3;
      cursor: wait;
      transform: none;
    }
    input {
      width: 66px;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 10px;
      font: inherit;
      outline: none;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 18px;
      box-shadow: var(--shadow);
      overflow: hidden;
    }
    .panel::before {
      content: "";
      display: block;
      height: 5px;
      margin: -18px -18px 18px;
      background: linear-gradient(90deg, #0f766e, #2563eb, #f59e0b);
    }
    .side-panel {
      position: sticky;
      top: 18px;
    }
    .chat-panel {
      min-height: 0;
    }
    .chat-panel::before {
      background: linear-gradient(90deg, #2563eb, #7c3aed, #0f766e);
    }
    .samples {
      display: flex;
      gap: 8px;
      margin-top: 12px;
      flex-wrap: wrap;
    }
    .sample {
      background: #ffffff;
      color: #253144;
      text-align: center;
      font-weight: 650;
      padding: 9px 11px;
      border: 1px solid var(--soft-line);
      border-left: 0;
      border-radius: 999px;
      box-shadow: 0 5px 14px rgba(20, 31, 48, 0.04);
      font-size: 13px;
    }
    .sample:hover {
      background: #eff6ff;
      border-color: #bfdbfe;
      box-shadow: 0 10px 20px rgba(37, 99, 235, 0.10);
    }
    .sample.active {
      background: #dbeafe;
      border-color: #2563eb;
      color: #1e3a8a;
    }
    .subhead {
      margin-top: 18px;
      padding-top: 16px;
      border-top: 1px solid var(--soft-line);
    }
    .subhead.first {
      margin-top: 0;
      padding-top: 0;
      border-top: 0;
      margin-bottom: 12px;
    }
    .subhead h2 {
      margin-bottom: 6px;
    }
    .muted {
      font-size: 13px;
      color: #697487;
    }
    section {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 20px;
      margin-top: 14px;
      box-shadow: var(--shadow);
    }
    #output section:first-child {
      margin-top: 0;
    }
    #output section h2::before {
      content: "";
      display: inline-block;
      width: 9px;
      height: 9px;
      margin-right: 8px;
      border-radius: 999px;
      background: linear-gradient(135deg, var(--accent), var(--teal));
      vertical-align: 1px;
    }
    h2 {
      margin: 0 0 12px;
      font-size: 17px;
      letter-spacing: 0;
    }
    pre {
      white-space: pre-wrap;
      word-break: break-word;
      margin: 0;
      font: inherit;
      line-height: 1.55;
    }
    .source {
      border: 1px solid var(--soft-line);
      border-radius: 12px;
      padding: 12px;
      margin-top: 12px;
      background: linear-gradient(180deg, #ffffff, #f8fafc);
      border-left: 4px solid #2563eb;
    }
    .citation {
      font-weight: 700;
      color: #243044;
      margin-bottom: 6px;
    }
    .score {
      color: #697487;
      font-weight: 500;
      font-size: 13px;
    }
    .error {
      border-color: #f4b8b2;
      background: #fff7f6;
      color: var(--danger);
    }
    .empty {
      min-height: 280px;
      display: grid;
      align-content: center;
      color: #697487;
      text-align: center;
      background:
        linear-gradient(135deg, rgba(15, 118, 110, 0.08), rgba(37, 99, 235, 0.08)),
        #ffffff;
      position: relative;
      overflow: hidden;
    }
    .empty::before {
      content: "";
      width: 120px;
      height: 120px;
      border-radius: 24px;
      border: 1px solid rgba(15, 118, 110, 0.22);
      position: absolute;
      left: 48px;
      top: 42px;
      transform: rotate(12deg);
    }
    .empty::after {
      content: "";
      width: 150px;
      height: 150px;
      border-radius: 50%;
      border: 1px solid rgba(245, 158, 11, 0.26);
      position: absolute;
      right: 60px;
      bottom: 42px;
    }
    .empty-content {
      position: relative;
      z-index: 1;
      max-width: 520px;
      margin: 0 auto;
    }
    .empty-title {
      display: block;
      color: var(--ink);
      font-size: 22px;
      font-weight: 800;
      margin-bottom: 8px;
    }
    .empty-steps {
      display: flex;
      gap: 8px;
      justify-content: center;
      flex-wrap: wrap;
      margin-top: 18px;
    }
    .empty-steps span {
      background: rgba(255, 255, 255, 0.78);
      border: 1px solid rgba(216, 222, 232, 0.88);
      border-radius: 999px;
      padding: 8px 10px;
      color: #40506a;
      font-size: 13px;
      font-weight: 700;
    }
    .file-row {
      background: linear-gradient(180deg, #f8fafc, #ffffff);
      border: 1px solid var(--soft-line);
      border-radius: 12px;
      padding: 12px;
    }
    .metric-row {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 10px;
      margin-bottom: 16px;
    }
    .metric {
      background: linear-gradient(180deg, #ffffff, #f8fafc);
      border: 1px solid var(--soft-line);
      border-radius: 14px;
      padding: 13px;
    }
    .metric strong {
      display: block;
      color: var(--ink);
      font-size: 18px;
      margin-bottom: 2px;
    }
    .metric span {
      color: var(--muted);
      font-size: 12px;
      font-weight: 700;
    }
    .upload-note {
      margin: 10px 0 14px;
      padding: 11px;
      border-radius: 12px;
      background: #eff6ff;
      color: #1e3a8a;
      font-size: 13px;
      font-weight: 650;
    }
    @media (max-width: 820px) {
      header {
        display: block;
      }
      .status {
        margin-top: 12px;
        justify-content: flex-start;
      }
      .layout {
        grid-template-columns: 1fr;
      }
      .side-panel {
        position: static;
      }
      .metric-row {
        grid-template-columns: 1fr;
      }
    }
  </style>
</head>
<body>
  <main>
    <header>
      <div>
        <div class="brand-kicker">Retrieval-Augmented Generation</div>
        <h1>RAG Document Q&A Bot</h1>
        <p>Grounded answers from your uploaded document collection.</p>
      </div>
      <div class="status">
        <span><strong>Vector store:</strong> storage/</span>
        <span><strong>Documents:</strong> PDF and TXT corpus</span>
        <span><strong>Mode:</strong> grounded answers with citations</span>
      </div>
    </header>

    <div class="layout">
      <div class="panel side-panel">
        <div class="subhead first">
          <h2>Knowledge Base</h2>
        </div>
        <div class="metric-row">
          <div class="metric"><strong>5</strong><span>Docs</span></div>
          <div class="metric"><strong>FAISS</strong><span>Store</span></div>
          <div class="metric"><strong>Top-K</strong><span>Search</span></div>
        </div>
        <div class="upload-note">Upload a file or paste text, then rebuild the index before asking about new content.</div>
        <form id="upload-form">
          <div class="file-row">
            <label class="field-label" for="document-file">Upload File</label>
            <input id="document-file" type="file" accept=".pdf,.txt,.md,.docx">
            <label class="field-label" for="document-title">Filename for Pasted Text</label>
            <input id="document-title" type="text" placeholder="Filename for pasted text, e.g. notes.txt">
            <label class="field-label" for="document-text">Paste Document Text</label>
            <textarea id="document-text" placeholder="Paste document text here if you are not uploading a file..."></textarea>
          </div>
          <div class="controls">
            <button id="upload-button" type="submit">Add and Reindex</button>
          </div>
        </form>
      </div>

      <div class="workspace">
        <div class="panel chat-panel">
          <div class="subhead first">
            <h2>Ask Documents</h2>
            <p class="muted">Ask in natural language. The bot retrieves chunks and answers with source citations.</p>
          </div>
          <form id="qa-form">
            <label class="field-label" for="question">Question</label>
            <textarea id="question" placeholder="Ask a question from the document collection..." required></textarea>
            <div class="controls">
              <button id="ask-button" type="submit">Ask Question</button>
              <label>Top K <input id="top-k" type="number" min="1" max="10" value="4"></label>
            </div>
          </form>
          <div class="samples">
            <button class="sample" type="button">How can urban tree canopy reduce heat risk in cities?</button>
            <button class="sample" type="button">What are the main reliability challenges for renewable microgrids?</button>
            <button class="sample" type="button">How is machine learning used in financial fraud detection?</button>
            <button class="sample" type="button">What privacy practices should small businesses follow?</button>
            <button class="sample" type="button">What are the risks of using AI in healthcare?</button>
            <button class="sample" type="button">What is the author's favorite movie?</button>
          </div>
        </div>

        <div id="output">
          <section class="empty">
            <div class="empty-content">
              <span class="empty-title">Ready for a grounded answer</span>
              <pre>Ask a question to see the generated answer, retrieved chunks, citations, and similarity scores.</pre>
              <div class="empty-steps">
                <span>Retrieve</span>
                <span>Generate</span>
                <span>Cite Sources</span>
              </div>
            </div>
          </section>
        </div>
      </div>
    </div>
  </main>

  <script>
    const form = document.getElementById("qa-form");
    const button = document.getElementById("ask-button");
    const uploadForm = document.getElementById("upload-form");
    const uploadButton = document.getElementById("upload-button");
    const output = document.getElementById("output");
    const questionInput = document.getElementById("question");

    function escapeHtml(value) {
      return value.replace(/[&<>"']/g, char => ({
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        '"': "&quot;",
        "'": "&#039;"
      }[char]));
    }

    document.querySelectorAll(".sample").forEach(sampleButton => {
      sampleButton.addEventListener("click", () => {
        document.querySelectorAll(".sample").forEach(item => item.classList.remove("active"));
        sampleButton.classList.add("active");
        questionInput.value = sampleButton.textContent;
        questionInput.focus();
      });
    });

    function readFileAsDataUrl(file) {
      return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(reader.result);
        reader.onerror = () => reject(reader.error);
        reader.readAsDataURL(file);
      });
    }

    uploadForm.addEventListener("submit", async event => {
      event.preventDefault();
      uploadButton.disabled = true;
      output.innerHTML = "<section><h2>Indexing Documents</h2><pre>Saving the document, chunking text, embedding chunks, and rebuilding FAISS...</pre></section>";

      const file = document.getElementById("document-file").files[0];
      const payload = {
        title: document.getElementById("document-title").value,
        text: document.getElementById("document-text").value
      };

      if (file) {
        payload.filename = file.name;
        payload.data_url = await readFileAsDataUrl(file);
      }

      try {
        const response = await fetch("/upload", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify(payload)
        });
        const data = await response.json();
        if (!response.ok) {
          throw new Error(data.error || "Upload failed");
        }
        output.innerHTML = `<section><h2>Index Updated</h2><pre>${escapeHtml(data.message)}</pre></section>`;
        uploadForm.reset();
      } catch (error) {
        output.innerHTML = `<section class="error"><h2>Upload Error</h2><pre>${escapeHtml(error.message)}</pre></section>`;
      } finally {
        uploadButton.disabled = false;
      }
    });

    form.addEventListener("submit", async event => {
      event.preventDefault();
      button.disabled = true;
      output.innerHTML = "<section><h2>Generating Answer</h2><pre>Searching the vector store, selecting source chunks, and asking the language model...</pre></section>";

      const payload = {
        question: document.getElementById("question").value,
        top_k: Number(document.getElementById("top-k").value || 4)
      };

      try {
        const response = await fetch("/ask", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify(payload)
        });
        const data = await response.json();
        if (!response.ok) {
          throw new Error(data.error || "Request failed");
        }

        const sources = data.sources.map(source => `
          <div class="source">
            <div class="citation">${escapeHtml(source.citation)} <span class="score">score=${source.score.toFixed(3)}</span></div>
            <pre>${escapeHtml(source.preview)}</pre>
          </div>
        `).join("");

        output.innerHTML = `
          <section>
            <h2>Answer</h2>
            <pre>${escapeHtml(data.answer)}</pre>
          </section>
          <section>
            <h2>Retrieved Sources</h2>
            ${sources}
          </section>
        `;
      } catch (error) {
        output.innerHTML = `<section class="error"><h2>Error</h2><pre>${escapeHtml(error.message)}</pre></section>`;
      } finally {
        button.disabled = false;
      }
    });
  </script>
</body>
</html>
"""


class RagWebApp:
    def __init__(self, data_dir: Path, storage_dir: Path) -> None:
        settings = load_settings()
        self.data_dir = data_dir
        self.storage_dir = storage_dir
        self.store = FaissVectorStore(storage_dir)
        self.store.load()
        self.embedding_model = EmbeddingModel()
        self.embedding_model.load(storage_dir / "embedding_model.json")
        self.llm = LlmClient(settings)

    def add_document(self, payload: dict[str, object]) -> str:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        filename = _safe_filename(str(payload.get("filename") or payload.get("title") or "uploaded_document.txt"))
        extension = Path(filename).suffix.lower()

        if payload.get("data_url"):
            if extension not in SUPPORTED_EXTENSIONS:
                supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
                raise ValueError(f"Unsupported file type. Supported: {supported}")
            data_url = str(payload["data_url"])
            encoded = data_url.split(",", 1)[1] if "," in data_url else data_url
            (self.data_dir / filename).write_bytes(base64.b64decode(encoded))
        else:
            text = str(payload.get("text") or "").strip()
            if not text:
                raise ValueError("Upload a file or paste document text.")
            if extension not in {".txt", ".md"}:
                filename = f"{Path(filename).stem or 'uploaded_document'}.txt"
            (self.data_dir / filename).write_text(text, encoding="utf-8")

        chunk_count = self.rebuild_index()
        return f"Saved {filename} and rebuilt the index with {chunk_count} chunks."

    def rebuild_index(self) -> int:
        pages = load_documents(self.data_dir)
        chunks = chunk_documents(pages)
        self.embedding_model = EmbeddingModel()
        embeddings = self.embedding_model.embed_texts([chunk.text for chunk in chunks])
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.embedding_model.save(self.storage_dir / "embedding_model.json")
        self.store.save(embeddings, chunks)
        self.store.load()
        return len(chunks)

    def ask(self, question: str, top_k: int) -> dict[str, object]:
        query_embedding = self.embedding_model.embed_query(question)
        results = self.store.search(query_embedding, top_k=top_k)
        answer = self.llm.generate(build_user_prompt(question, results))
        return {
            "answer": answer,
            "sources": [
                {
                    "citation": result.chunk.citation,
                    "score": result.score,
                    "preview": result.chunk.text[:700],
                }
                for result in results
            ],
        }


def create_handler(app: RagWebApp) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if urlparse(self.path).path != "/":
                self.send_error(404)
                return
            body = HTML_PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self) -> None:
            path = urlparse(self.path).path
            if path not in {"/ask", "/upload"}:
                self.send_error(404)
                return
            try:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length).decode("utf-8"))
                if path == "/upload":
                    self._send_json({"message": app.add_document(payload)})
                    return

                question = str(payload.get("question", "")).strip()
                top_k = int(payload.get("top_k", 4))
                if not question:
                    raise ValueError("Question is required.")
                response = app.ask(question, top_k=max(1, min(top_k, 10)))
                self._send_json(response)
            except Exception as exc:
                self._send_json({"error": str(exc)}, status=500)

        def log_message(self, format: str, *args: object) -> None:
            return

        def _send_json(self, payload: dict[str, object], status: int = 200) -> None:
            body = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    return Handler


def _safe_filename(filename: str) -> str:
    filename = Path(filename).name.strip()
    filename = re.sub(r"[^A-Za-z0-9._ -]+", "_", filename)
    filename = filename.strip(" .")
    return filename or "uploaded_document.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a simple web UI for the document Q&A bot.")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--storage-dir", default="storage")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    app = RagWebApp(Path(args.data_dir), Path(args.storage_dir))
    server = ThreadingHTTPServer((args.host, args.port), create_handler(app))
    print(f"Web app running at http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop.")
    server.serve_forever()


if __name__ == "__main__":
    main()
