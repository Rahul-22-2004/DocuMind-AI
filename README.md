# 📚 AI Document Assistant

AI-powered **Retrieval-Augmented Generation (RAG)** application that lets you **upload documents** (PDF, text, images) and **chat** with them to get **source-backed answers**.
Built with **Flask** (backend) + **Streamlit Premium UI** (frontend) + **FAISS** (vector database) + **Gemini** (LLM).

---

## **✨ Features**

- 📂 **File Uploads** → Supports PDFs, TXT, Markdown, and images (PNG, JPG, TIFF).
- 🧠 **RAG-Powered Chat** → Ask questions & get AI-generated answers from your uploaded documents.
- 🔍 **Semantic Search** → Uses **FAISS** + **Google Gemini embeddings** for relevant retrieval.
- 🖼️ **Automatic OCR** → Extracts text from scanned PDFs and images using **Tesseract**.
- 📌 **Source Citations** → Every answer includes references to document names & page numbers.
- ⚡ **Streamlit Premium UI** → Modern chat interface with:

  - WhatsApp-style chat bubbles
  - Avatars & timestamps
  - Collapsible citations panel
  - Dark mode support

---

## **🛠️ Tech Stack**

| Layer          | Technology                                  |
| -------------- | ------------------------------------------- |
| **Frontend**   | Streamlit (Premium UI)                      |
| **Backend**    | Flask (Python)                              |
| **Vector DB**  | FAISS                                       |
| **LLM**        | Google Gemini / MiniLM                      |
| **OCR**        | Tesseract                                   |
| **Embeddings** | Google Generative AI / SentenceTransformers |

---

## **📦 Project Structure**

```
AI-DOCUMENT-ASSISTANT/
│── app.py                # Flask backend (RAG logic, OCR, embeddings, FAISS)
│── streamlit_app.py      # Streamlit premium frontend
│── requirements.txt      # Python dependencies
│── input_data/           # Uploaded files and vector indexes
│── README.md             # Documentation
│── .env                  # Environment variables (API keys, config)
└── .venv/                # Virtual environment (optional)
```

---

## **⚡ Setup & Installation**

### **1. Clone the Repository**

```bash
git clone https://github.com/Rahul-22-2004/DocuMind-AI.git
cd ai-document-assistant
```

### **2. Create & Activate Virtual Environment** _(optional but recommended)_

```bash
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Mac/Linux:
source .venv/bin/activate
```

### **3. Install Dependencies**

```bash
pip install -r requirements.txt
```

### **4. Configure Environment Variables**

Create a **`.env`** file in the project root and add:

```env
GOOGLE_API_KEY=your_google_api_key
EMBEDDING_BACKEND=gemini   # or minilm
TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe
PORT=8000
FE_PORT=8501
```

---

## **🚀 Running the Application**

We need **two terminals** — one for the **Flask backend** and one for the **Streamlit frontend**.

### **1. Start Flask Backend**

```bash
python app.py
```

- Backend runs at: **[http://localhost:8000](http://localhost:8000)**

### **2. Start Streamlit Frontend**

```bash
streamlit run streamlit_app.py
```

- Streamlit UI opens at: **[http://localhost:8501](http://localhost:8501)**

---

## **🧩 How to Use**

### **Step 1 → Check Backend Status**

- On the Streamlit UI, the **Backend Status** panel will confirm the connection.

### **Step 2 → Upload Files**

- Go to the **📂 Upload & Index** section.
- Drag & drop **PDFs, TXT, Markdown, or Images**.
- Click **📑 Index Documents** or **🖼️ Index Images**.

### **Step 3 → Chat with Your Documents**

- Navigate to the **💬 Chat** section.
- Ask a question → click **Send** or press **Enter**.
- Get **AI-generated answers** + **citations**.

---

## **📌 Citations Example**

> **Answer:**
> The loan repayment period is **24 months**.

**Citations:**

- 📄 `policy_doc` (p.5) — policy.pdf
- 📄 `guidelines` (p.2) — loan_rules.pdf

---

## **⚙️ Top-K Documents to Search**

- **Top-K** = Number of most relevant document chunks retrieved from FAISS.
- Default = `5` → balanced speed & accuracy.
- Adjust it using the slider in Streamlit UI:

  - `Top-K = 3` → Faster but may miss info.
  - `Top-K = 8` → More complete answers but slower.

---

## **🧠 RAG Workflow**

```
User Question → Convert to Embedding → Search FAISS → Retrieve Top-K Chunks →
Send Context + Question to Gemini → Generate Answer → Show with Citations
```

---

## **🧪 Testing API Endpoints**

### **Health Check**

```bash
curl http://localhost:8000/health
```

### **Upload Docs**

```bash
curl -X POST -F "files=@sample.pdf" http://localhost:8000/upload_docs
```

### **Chat**

```bash
curl -X POST http://localhost:8000/chat \
     -H "Content-Type: application/json" \
     -d '{"question": "What is the policy rate?"}'
```

---

## **🛠️ Troubleshooting**

| Issue                      | Fix                                               |
| -------------------------- | ------------------------------------------------- |
| **Tesseract not found**    | Update `TESSERACT_CMD` in `.env`                  |
| **Backend not connecting** | Ensure Flask is running on **port 8000**          |
| **No answers returned**    | Increase **Top-K** slider                         |
| **Slow responses**         | Reduce Top-K or switch `EMBEDDING_BACKEND=minilm` |

---

## **🌟 Future Enhancements**

- ✅ Real-time streaming answers (ChatGPT-style typing effect)
- ✅ Highlighting relevant text passages
- ✅ Save chat history per session
- ✅ Deploy on **AWS / GCP / Azure**

---

## **🚀 Quick Start**

```bash
# Start backend
python app.py

# Start frontend
streamlit run streamlit_app.py
```

Then open → **[http://localhost:8501](http://localhost:8501)**

---

## **🔗 Connect with Me**

📧 Email: [rahuldgowda2004@example.com]
