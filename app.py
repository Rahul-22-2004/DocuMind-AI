# app.py
import io
import os
import threading
import uuid
from typing import Any, Dict, List, Optional, Tuple
import fitz  # PyMuPDF
import google.genai as genai
from google.genai import types
import pytesseract
from dotenv import load_dotenv
from flask import Flask, jsonify, request
from flask_cors import CORS
from langchain.embeddings.base import Embeddings
from langchain_text_splitters import \
    RecursiveCharacterTextSplitter  # (default text splitter)
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_experimental.text_splitter import \
    SemanticChunker  # (optional, for semantic chunking)
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from PIL import Image

# =============================================================================
# App and Global Configuration
# =============================================================================

load_dotenv()
app = Flask(__name__)
CORS(app)

# --- Paths ---
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(BASE_DIR, "input_data")
UPLOAD_DIR = os.path.join(DATA_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# --- Embedding Backend Setting ---
EMBEDDING_BACKEND = os.getenv("EMBEDDING_BACKEND", "gemini").strip().lower()  # "minilm" or "gemini"
VECTOR_DIR = os.path.join(DATA_DIR, f"faiss_{EMBEDDING_BACKEND}")
os.makedirs(VECTOR_DIR, exist_ok=True)

# --- Gemini API Key for Chat and (optional) Embeddings ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY is required.")

# Create a global client for the new GenAI SDK (replaces genai.configure)
client = genai.Client(api_key=GOOGLE_API_KEY)

# --- Optional Tesseract Path for Windows ---
TESSERACT_CMD = os.getenv("TESSERACT_CMD", r"C:\Program Files\Tesseract-OCR\tesseract.exe")
if TESSERACT_CMD:
    pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

# --- Chunking Defaults ---
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))
CHUNKING_MODE_DEFAULT = os.getenv("CHUNKING_MODE", "semantic").strip().lower()  # "recursive" or "semantic"

# --- SemanticChunker tuning (used only when CHUNKING_MODE=semantic) ---
SEMANTIC_BREAKPOINT_TYPE = os.getenv("SEMANTIC_BREAKPOINT_TYPE", "percentile").strip().lower()
# Typically 95 works well for "percentile" type
try:
    SEMANTIC_BREAKPOINT_AMOUNT = float(os.getenv("SEMANTIC_BREAKPOINT_AMOUNT", "95"))
except ValueError:
    SEMANTIC_BREAKPOINT_AMOUNT = 95.0

# --- Chat Defaults ---
TOP_K_DEFAULT = int(os.getenv("TOP_K", "5"))

# --- Thread-safety for Vector Store Updates ---
INDEX_LOCK = threading.Lock()

# --- In-memory Chat Sessions: session_id -> list of {role, content} ---
CHAT_SESSIONS: Dict[str, List[Dict[str, str]]] = {}

# --- Global Embeddings Instance (lazy init) ---
_EMBEDDINGS: Optional[Embeddings] = None


# =============================================================================
# Utility and Core Functions
# =============================================================================

def get_embeddings() -> Embeddings:
    """
    Return a shared embeddings instance based on EMBEDDING_BACKEND.
    - "minilm": sentence-transformers/all-MiniLM-L6-v2 (CPU-friendly)
    - "gemini": Google text-embedding-004 (requires GOOGLE_API_KEY)
    """
    global _EMBEDDINGS
    if _EMBEDDINGS is not None:
        return _EMBEDDINGS

    if EMBEDDING_BACKEND == "gemini":
        if not GOOGLE_API_KEY:
            raise RuntimeError("GOOGLE_API_KEY is required for Gemini embeddings.")
        _EMBEDDINGS = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    else:
        # Default local embedding
        _EMBEDDINGS = SentenceTransformerEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return _EMBEDDINGS


def load_vector_store(allow_create_empty: bool = True) -> Optional[FAISS]:
    """
    Load the FAISS vector store from disk. If not found and allow_create_empty is False,
    returns None. Otherwise, returns a ready-to-use FAISS store (or None if empty).
    """
    embeddings = get_embeddings()
    index_file = os.path.join(VECTOR_DIR, "index.faiss")
    store_file = os.path.join(VECTOR_DIR, "index.pkl")

    if os.path.exists(index_file) and os.path.exists(store_file):
        # allow_dangerous_deserialization required when loading pickled docstore
        return FAISS.load_local(VECTOR_DIR, embeddings, allow_dangerous_deserialization=True)

    if not allow_create_empty:
        return None
    return None


def save_vector_store(store: FAISS) -> None:
    """
    Persist the FAISS vector store to disk at VECTOR_DIR.
    """
    store.save_local(VECTOR_DIR)


def build_text_splitter(
    mode: str = CHUNKING_MODE_DEFAULT,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
):
    """
    Build and return a text splitter based on the selected mode.
    - recursive: RecursiveCharacterTextSplitter
    - semantic: SemanticChunker (requires embeddings)
    """
    mode = (mode or "recursive").strip().lower()
    if mode == "semantic":
        embeddings = get_embeddings()
        return SemanticChunker(
            embeddings=embeddings,
            breakpoint_threshold_type=SEMANTIC_BREAKPOINT_TYPE,
            breakpoint_threshold_amount=SEMANTIC_BREAKPOINT_AMOUNT,
        )
    # default fallback: recursive
    return RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)


def chunk_documents(
    docs: List[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    chunking_mode: str = CHUNKING_MODE_DEFAULT,
) -> List[Document]:
    """
    Split a list of Documents into chunks using the configured splitter.
    Supports:
    - recursive (default)
    - semantic (optional)
    """
    splitter = build_text_splitter(mode=chunking_mode, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(docs)


def add_documents_to_index(
    docs: List[Document],
    chunking_mode: str = CHUNKING_MODE_DEFAULT,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> Tuple[int, int]:
    """
    Add Documents to FAISS index (create if needed), then persist.
    Returns (num_original_docs, num_chunks_added).
    """
    if not docs:
        return 0, 0

    # Chunk first
    chunks = chunk_documents(docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap, chunking_mode=chunking_mode)

    embeddings = get_embeddings()

    with INDEX_LOCK:
        store = load_vector_store(allow_create_empty=True)
        if store is None:
            store = FAISS.from_documents(chunks, embeddings)
        else:
            store.add_documents(chunks)
        save_vector_store(store)

    return len(docs), len(chunks)


def pdf_to_documents(pdf_path: str, source_id: Optional[str] = None) -> List[Document]:
    """
    Extract text from a PDF using PyPDFLoader (one Document per page).
    Adds simple metadata for citations (source, page).
    """
    try:
        loader = PyPDFLoader(pdf_path)
        pages = loader.load()
        # Attach/standardize metadata
        for i, d in enumerate(pages):
            d.metadata = d.metadata or {}
            d.metadata.update({
                "source": os.path.basename(pdf_path),
                "doc_id": source_id or os.path.splitext(os.path.basename(pdf_path))[0],
                "page": d.metadata.get("page", i + 1)
            })
        return pages
    except Exception as e:
        print(f"[WARN] PyPDFLoader failed on {pdf_path}: {e}")
        return []


def ocr_image_bytes(image_bytes: bytes, source_name: str) -> Document:
    """
    OCR a single image (bytes) into a Document with metadata for citation.
    """
    img = Image.open(io.BytesIO(image_bytes))
    text = pytesseract.image_to_string(img)
    return Document(
        page_content=text.strip(),
        metadata={
            "source": source_name,
            "doc_id": os.path.splitext(os.path.basename(source_name))[0],
            "page": None,
            "kind": "image_ocr"
        }
    )


def ocr_pdf_pages(pdf_path: str, dpi: int = 200) -> List[Document]:
    """
    OCR a PDF by rasterizing each page and running Tesseract OCR.
    Use when text extraction fails (e.g., scanned PDFs).
    """
    out_docs: List[Document] = []
    try:
        with fitz.open(pdf_path) as doc:
            for page_index in range(doc.page_count):
                page = doc.load_page(page_index)
                mat = fitz.Matrix(dpi / 72, dpi / 72)
                pix = page.get_pixmap(matrix=mat, alpha=False)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                text = pytesseract.image_to_string(img)
                out_docs.append(Document(
                    page_content=text.strip(),
                    metadata={
                        "source": os.path.basename(pdf_path),
                        "doc_id": os.path.splitext(os.path.basename(pdf_path))[0],
                        "page": page_index + 1,
                        "kind": "pdf_ocr"
                    }
                ))
    except Exception as e:
        print(f"[ERROR] OCR on PDF failed for {pdf_path}: {e}")
    return out_docs


def txt_to_document(text_bytes: bytes, filename: str) -> Document:
    """
    Convert a text file (bytes) to a single Document.
    """
    try:
        content = text_bytes.decode("utf-8", errors="ignore")
    except UnicodeDecodeError:
        content = text_bytes.decode("latin-1", errors="ignore")
    return Document(
        page_content=content.strip(),
        metadata={
            "source": filename,
            "doc_id": os.path.splitext(os.path.basename(filename))[0],
            "page": None,
            "kind": "text"
        }
    )


def ensure_session(session_id: Optional[str]) -> str:
    """
    Ensure a session_id exists; return an existing or newly created UUID.
    """
    if session_id and session_id in CHAT_SESSIONS:
        return session_id
    sid = session_id or str(uuid.uuid4())
    CHAT_SESSIONS.setdefault(sid, [])
    return sid


def messages_to_history_str(messages: List[Dict[str, str]], max_turns: int = 6) -> str:
    """
    Convert recent messages into a string for question condensation prompt.
    """
    recent = messages[-(2 * max_turns):] if messages else []
    lines = []
    for m in recent:
        role = m.get("role", "user")
        content = m.get("content", "")
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def format_docs_for_context(docs: List[Document]) -> str:
    """
    Create a compact context string from retrieved documents, with citations.
    """
    lines = []
    for d in docs:
        m = d.metadata or {}
        cite = f"{m.get('doc_id', '?')} p.{m.get('page', '?')}"
        lines.append(f"[{cite}] {d.page_content}")
    return "\n\n".join(lines)


def retrieve_top_k(query: str, k: int = TOP_K_DEFAULT) -> List[Document]:
    """
    Retrieve top-k relevant Documents from FAISS using the configured embeddings.
    """
    store = load_vector_store(allow_create_empty=False)
    if store is None:
        return []
    retriever = store.as_retriever(search_type="similarity", search_kwargs={"k": k})
    return retriever.invoke(query)


def condense_question(question: str, session_id: str) -> str:
    """
    Condense the question using Gemini, considering chat history.
    Fallback to original question on error.
    """
    try:
        history = CHAT_SESSIONS.get(session_id, [])
        if not history:
            return question  # No history, no need to condense

        # Build prompt with history (example; adjust your prompt as needed)
        prompt = f"Given this chat history: {history}\n\nCondense this follow-up question into a standalone question: {question}"

        response = client.models.generate_content(
            model="gemini-flash-latest",  # Use a valid, fast model
            contents=[{"role": "user", "parts": [{"text": prompt}]}],
            config=types.GenerateContentConfig(temperature=0.0)  # Low temp for factual tasks
        )
        return response.candidates[0].content.parts[0].text.strip()
    except Exception as e:
        print(f"[WARN] condense_question failed: {e}")
        return question

def answer_with_gemini(standalone_q: str, docs: List[Document]) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Generate an answer using Gemini with retrieved docs as context.
    Returns (answer, citations).
    """
    try:
        if not docs:
            return "I don't have enough information from the uploaded documents to answer this question.", []

        # Build clean context
        context_parts = []
        citations = []
        for i, doc in enumerate(docs):
            source = doc.metadata.get("source", "Unknown document")
            page = doc.metadata.get("page", "unknown")
            context_parts.append(f"From {source} (page {page}):\n{doc.page_content.strip()}")
            
            citations.append({
                "doc_id": i + 1,
                "source": source,
                "page": page,
                "kind": "text"
            })

        context = "\n\n".join(context_parts)

        prompt = f"""You are an expert assistant that answers questions STRICTLY based ONLY on the provided context from the uploaded document. 

Context:
{context}

Question: {standalone_q}

Rules:
- Use ONLY information explicitly present in the context above.
- If the context does not contain enough information to answer fully or accurately, say "The uploaded document does not provide sufficient details on this specific point."
- Do NOT add general knowledge or information not directly stated in the context.
- Answer clearly and naturally, using bullet points or numbered lists when appropriate.
- List exactly the scopes/points mentioned in the context.
- At the end, add: "Sources: Based on {len(docs)} retrieved section(s) from the uploaded document."

Answer:"""

        response = client.models.generate_content(
            model="gemini-flash-latest",
            contents=[{"role": "user", "parts": [{"text": prompt}]}],
            config=types.GenerateContentConfig(
                temperature=0.3,
                max_output_tokens=1024
            )
        )
        answer = response.candidates[0].content.parts[0].text.strip()

        # Optional: append a clean source note if not in response
        if "Sources:" not in answer:
            answer += f"\n\nSources: Based on {len(citations)} page(s) from the uploaded document(s)."

        return answer, citations
    except Exception as e:
        print(f"[ERROR] answer_with_gemini failed: {e}")
        return "Sorry, I encountered an error while generating the answer.", []

# =============================================================================
# Flask Endpoints
# =============================================================================

@app.route("/health", methods=["GET"])
def health():
    """
    Health check endpoint.
    """
    return jsonify({
        "status": "ok",
        "embedding_backend": EMBEDDING_BACKEND,
        "default_chunking_mode": CHUNKING_MODE_DEFAULT,
        "semantic_breakpoint_type": SEMANTIC_BREAKPOINT_TYPE,
        "semantic_breakpoint_amount": SEMANTIC_BREAKPOINT_AMOUNT,
    })


@app.route("/stats", methods=["GET"])
def stats():
    """
    Return basic stats about the current FAISS store.
    """
    store = load_vector_store(allow_create_empty=True)
    count = 0
    if store is not None and hasattr(store, "docstore") and hasattr(store.docstore, "_dict"):
        count = len(store.docstore._dict)

    return jsonify({
        "vector_dir": VECTOR_DIR,
        "embedding_backend": EMBEDDING_BACKEND,
        "documents_indexed": count,
        "default_chunking_mode": CHUNKING_MODE_DEFAULT,
        "semantic_breakpoint_type": SEMANTIC_BREAKPOINT_TYPE,
        "semantic_breakpoint_amount": SEMANTIC_BREAKPOINT_AMOUNT,
    })


@app.route("/upload_docs", methods=["POST"])
def upload_docs():
    """
    Upload and index document files (PDF/TXT). Accepts multiple files via 'files'.
    For PDFs: extract text per page; fallback to OCR if needed.
    For TXT: load file as a single document.
    Optional multipart form fields:
    - chunking or chunking_mode: "recursive" (default) or "semantic"
    - chunk_size: int (for recursive mode only)
    - chunk_overlap: int (for recursive mode only)
    """
    if "files" not in request.files:
        return jsonify({"error": "No files part in request. Use form-data with 'files'."}), 400

    files = request.files.getlist("files")
    if not files:
        return jsonify({"error": "No files uploaded."}), 400

    # Read optional chunking controls from form
    chunking_mode = (request.form.get("chunking") or request.form.get("chunking_mode") or CHUNKING_MODE_DEFAULT).strip().lower()
    try:
        chunk_size = int(request.form.get("chunk_size", CHUNK_SIZE))
    except Exception:
        chunk_size = CHUNK_SIZE
    try:
        chunk_overlap = int(request.form.get("chunk_overlap", CHUNK_OVERLAP))
    except Exception:
        chunk_overlap = CHUNK_OVERLAP

    processed: List[Document] = []
    num_files = len(files)

    for f in files:
        filename = f.filename or f"file_{uuid.uuid4().hex}"
        ext = os.path.splitext(filename)[1].lower()

        # Save to disk first (needed by PyPDFLoader)
        dest_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4().hex}_{filename}")
        f.save(dest_path)

        try:
            if ext == ".pdf":
                docs = pdf_to_documents(dest_path)
                # Fallback OCR for scanned PDFs or empty extraction
                if not docs or sum(len(d.page_content or "") for d in docs) < 50:
                    docs = ocr_pdf_pages(dest_path, dpi=200)
                processed.extend(docs)
            elif ext in (".txt", ".md"):
                # Read text, build Document
                with open(dest_path, "rb") as rf:
                    text_bytes = rf.read()
                processed.append(txt_to_document(text_bytes, filename))
            else:
                # Unsupported in this endpoint
                pass
        except Exception as e:
            print(f"[ERROR] Failed processing {filename}: {e}")

    d_count, c_count = add_documents_to_index(
        processed,
        chunking_mode=chunking_mode,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    return jsonify({
        "message": "Documents indexed.",
        "files_processed": num_files,
        "original_docs": d_count,
        "chunks_added": c_count,
        "chunking_mode_used": chunking_mode
    })


@app.route("/upload_images", methods=["POST"])
def upload_images():
    """
    Upload and index image files (JPG/PNG/TIFF). Accepts multiple files via 'files'.
    Each image is OCR'd with Tesseract and added to the vector store.
    Optional multipart form fields:
    - chunking or chunking_mode: "recursive" (default) or "semantic"
    - chunk_size: int (for recursive mode only)
    - chunk_overlap: int (for recursive mode only)
    """
    if "files" not in request.files:
        return jsonify({"error": "No files part in request. Use form-data with 'files'."}), 400

    files = request.files.getlist("files")
    if not files:
        return jsonify({"error": "No files uploaded."}), 400

    # Read optional chunking controls from form
    chunking_mode = (request.form.get("chunking") or request.form.get("chunking_mode") or CHUNKING_MODE_DEFAULT).strip().lower()
    try:
        chunk_size = int(request.form.get("chunk_size", CHUNK_SIZE))
    except Exception:
        chunk_size = CHUNK_SIZE
    try:
        chunk_overlap = int(request.form.get("chunk_overlap", CHUNK_OVERLAP))
    except Exception:
        chunk_overlap = CHUNK_OVERLAP

    processed: List[Document] = []
    num_files = len(files)

    for f in files:
        filename = f.filename or f"image_{uuid.uuid4().hex}.png"
        ext = os.path.splitext(filename)[1].lower()
        if ext not in (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"):
            continue
        try:
            img_bytes = f.read()
            doc = ocr_image_bytes(img_bytes, filename)
            if doc.page_content:
                processed.append(doc)
        except Exception as e:
            print(f"[ERROR] OCR failed for {filename}: {e}")

    d_count, c_count = add_documents_to_index(
        processed,
        chunking_mode=chunking_mode,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    return jsonify({
        "message": "Images OCR'd and indexed.",
        "files_processed": num_files,
        "original_docs": d_count,
        "chunks_added": c_count,
        "chunking_mode_used": chunking_mode
    })


@app.route("/chat", methods=["POST"])
def chat():
    """
    Chat with the vector-backed RAG assistant.
    Body JSON:
    - question: str (required)
    - session_id: str (optional; for memory; new one will be created if missing)
    - top_k: int (optional; default TOP_K_DEFAULT)
    Returns:
    - session_id
    - question
    - answer
    - citations: list of {doc_id, page, source, kind}
    """
    data = request.get_json(force=True, silent=True) or {}
    question = (data.get("question") or "").strip()
    session_id = data.get("session_id")
    top_k = int(data.get("top_k", TOP_K_DEFAULT))
    if not question:
        return jsonify({"error": "Missing 'question' in request body."}), 400

    sid = ensure_session(session_id)

    # Check store
    store = load_vector_store(allow_create_empty=False)
    if store is None:
        return jsonify({"error": "Vector store empty. Upload documents/images first."}), 400

    # Memory: add user turn first (so condensation sees it too)
    CHAT_SESSIONS[sid].append({"role": "user", "content": question})

    # Condense follow-up to standalone
    standalone_q = condense_question(question, sid)

    # Retrieve
    docs = retrieve_top_k(standalone_q, k=top_k)

    # Answer with Gemini
    try:
        answer, citations = answer_with_gemini(standalone_q, docs)
    except Exception as e:
        # On failure, remove the just-added user turn to keep memory clean
        CHAT_SESSIONS[sid].pop()
        return jsonify({"error": f"LLM error: {e}"}), 500

    # Save assistant turn
    CHAT_SESSIONS[sid].append({"role": "assistant", "content": answer})

    return jsonify({
        "session_id": sid,
        "question": question,
        "standalone_question": standalone_q,
        "answer": answer,
        "citations": citations
    })


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    # --- Helpful prints on startup ---
    print(f"Embedding backend: {EMBEDDING_BACKEND}")
    print(f"Default chunking mode: {CHUNKING_MODE_DEFAULT}")
    print(f"Semantic breakpoint: type={SEMANTIC_BREAKPOINT_TYPE}, amount={SEMANTIC_BREAKPOINT_AMOUNT}")
    if EMBEDDING_BACKEND == "gemini" and not GOOGLE_API_KEY:
        print("[WARN] EMBEDDING_BACKEND=gemini but GOOGLE_API_KEY is missing. Embeddings will fail.")
    if not GOOGLE_API_KEY:
        print("[WARN] GOOGLE_API_KEY missing. Chat and question condensation will be disabled.")

    try:
        # Verify tesseract availability
        ver = pytesseract.get_tesseract_version()
        print(f"Tesseract: {ver}")
    except Exception as e:
        print(f"[WARN] Tesseract not found or misconfigured: {e}")

    # --- Warm up embeddings ---
    try:
        _ = get_embeddings()
        print("Embeddings initialized.")
    except Exception as e:
        print(f"[WARN] Embeddings init failed: {e}")

    # Use $PORT if set (for Render), else 8000 locally
    port = int(os.environ.get("PORT", 8000))
    app.run(host="0.0.0.0", port=port, debug=False)  # Debug=False for prod safety