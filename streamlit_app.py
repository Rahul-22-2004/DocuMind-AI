import os
import requests
import streamlit as st
from datetime import datetime

# -------------------------
# CONFIGURATION
# -------------------------
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")
st.set_page_config(
    page_title="AI Document Assistant",
    page_icon="📚",
    layout="wide",
)

# -------------------------
# HELPER FUNCTIONS
# -------------------------
def ping_backend():
    try:
        r = requests.get(f"{BACKEND_URL}/health", timeout=5)
        if r.status_code == 200:
            data = r.json()
            return True, f"✅ Connected | Using `{data.get('embedding_backend', '?')}` embeddings"
        return False, "⚠️ Backend not responding."
    except Exception as e:
        return False, f"❌ Error: {e}"

def upload_files(files, endpoint="upload_docs"):
    if not files:
        st.warning("⚠️ Please select files to upload.")
        return None
    try:
        files_payload = [("files", (f.name, f, "application/octet-stream")) for f in files]
        with st.spinner("📤 Uploading files..."):
            r = requests.post(f"{BACKEND_URL}/{endpoint}", files=files_payload, timeout=120)
        if r.status_code != 200:
            st.error(f"❌ Upload failed: {r.text}")
            return None
        return r.json()
    except Exception as e:
        st.error(f"❌ Error: {e}")
        return None

def chat_with_backend(question, session_id, top_k=5):
    payload = {"question": question, "session_id": session_id, "top_k": top_k}
    try:
        r = requests.post(f"{BACKEND_URL}/chat", json=payload, timeout=60)
        if r.status_code == 200:
            return r.json()
        else:
            st.error(f"❌ Chat failed: {r.text}")
            return None
    except Exception as e:
        st.error(f"❌ Error: {e}")
        return None

# -------------------------
# CUSTOM CHAT BUBBLE RENDERER
# -------------------------
def render_chat(history):
    """Render chat history with WhatsApp-style bubbles."""
    for msg in history:
        is_user = msg["role"] == "user"
        bubble_color = "#2563eb" if is_user else "#22c55e"
        text_color = "#fff"
        avatar = "🧑‍💻" if is_user else "🤖"
        alignment = "flex-end" if is_user else "flex-start"
        timestamp = msg.get("time", datetime.now().strftime("%H:%M"))

        st.markdown(
            f"""
            <div style="
                display: flex;
                justify-content: {alignment};
                margin: 5px 0;
            ">
                <div style="
                    background-color: {bubble_color};
                    color: {text_color};
                    padding: 10px 14px;
                    border-radius: 12px;
                    max-width: 70%;
                    font-size: 15px;
                    box-shadow: 0px 2px 6px rgba(0,0,0,0.15);
                ">
                    <span style="font-size: 20px; margin-right: 5px;">{avatar}</span>
                    {msg['content']}
                    <div style="font-size: 11px; color: #d1d5db; text-align: right; margin-top: 3px;">
                        {timestamp}
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

# -------------------------
# HEADER
# -------------------------
st.markdown(
    """
    <div style="background: linear-gradient(to right, #2563eb, #1e40af);
                padding: 15px;
                border-radius: 12px;
                text-align: center;
                margin-bottom: 20px;">
        <h1 style="color:white; margin:0;">📚 AI Document Assistant</h1>
        <p style="color:white; margin:0; font-size:14px;">
            Upload PDFs, images & text files → Chat with your documents → Get instant answers with citations.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# -------------------------
# BACKEND STATUS
# -------------------------
status_ok, status_msg = ping_backend()
if status_ok:
    st.success(status_msg)
else:
    st.error(status_msg)

# -------------------------
# FILE UPLOAD & INDEXING
# -------------------------
st.markdown("### 📂 Upload and Index Documents")
col1, col2 = st.columns(2)

with col1:
    docs = st.file_uploader(
        "📄 Upload PDFs, TXT, Markdown",
        type=["pdf", "txt", "md"],
        accept_multiple_files=True,
    )
    if st.button("📑 Index Documents", use_container_width=True):
        result = upload_files(docs, "upload_docs")
        if result:
            st.success("✅ Documents Indexed Successfully!")
            st.json(result)

with col2:
    imgs = st.file_uploader(
        "🖼️ Upload Images for OCR (PNG/JPG/TIFF)",
        type=["png", "jpg", "jpeg", "tif", "tiff"],
        accept_multiple_files=True,
    )
    if st.button("🖼️ Index Images", use_container_width=True):
        result = upload_files(imgs, "upload_images")
        if result:
            st.success("✅ Images OCR'd and Indexed!")
            st.json(result)

# -------------------------
# CHAT SECTION
# -------------------------
st.markdown("---")
st.subheader("💬 Chat with Your Documents")

# Session state for chat
if "session_id" not in st.session_state:
    st.session_state.session_id = os.urandom(8).hex()
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Chat interface
top_k = st.slider("Top-K Documents to Search", 1, 10, 5)
question = st.text_input("Ask a question...", placeholder="Type your question here and press Enter")

if st.button("🚀 Send Query", use_container_width=True):
    if not question.strip():
        st.warning("⚠️ Please enter a valid question.")
    else:
        response = chat_with_backend(question, st.session_state.session_id, top_k)
        if response:
            st.session_state.chat_history.append({
                "role": "user",
                "content": question,
                "time": datetime.now().strftime("%H:%M")
            })
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": response["answer"],
                "time": datetime.now().strftime("%H:%M")
            })

            # Render chat history
            render_chat(st.session_state.chat_history)

            # Citations display
            with st.expander("📌 Citations", expanded=False):
                if response.get("citations"):
                    for c in response["citations"]:
                        st.markdown(
                            f"- **{c['doc_id']}** (p.{c['page']}) — *{c['source']}*"
                        )
                else:
                    st.info("No citations found.")

# -------------------------
# FOOTER
# -------------------------
st.markdown("---")
st.caption("⚡ Built with Streamlit + Flask | AI-powered Document Assistant | Premium UI ✨")
