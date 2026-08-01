# DocuMind 🧠📄

DocuMind is an end-to-end Retrieval-Augmented Generation (RAG) web application that allows users to upload PDF documents and ask questions about their content in a conversational interface. 

The application uses state-of-the-art open-source LLMs through the Hugging Face Serverless Inference API to provide accurate, context-aware answers based strictly on the uploaded documents.

---

## 🚀 Features

- **PDF Ingestion & Processing**: Upload any PDF document to automatically extract and process its text.
- **Advanced Chunking**: Uses `RecursiveCharacterTextSplitter` with intelligent chunk overlapping to preserve context.
- **Local Vector Search**: Generates semantic embeddings locally (via `all-MiniLM-L6-v2`) and performs ultra-fast similarity search using Meta's **FAISS** in-memory vector database.
- **Serverless LLM Integration**: Dynamically routes queries to **Qwen2.5-72B-Instruct** (or similar available models) using the official `huggingface_hub` `InferenceClient`.
- **Interactive UI**: Clean, responsive chat interface built with **Streamlit**, complete with session state management for persistent conversation history.
- **Secure Credentials**: Uses `python-dotenv` for local, secure API key management.

---

## 🛠️ Tech Stack

- **Frontend & Backend Orchestration**: Python 3.13, Streamlit
- **RAG Framework**: LangChain (`langchain`, `langchain-community`)
- **Document Parsing**: PyPDF (`pypdf`)
- **Embeddings**: HuggingFace Embeddings (`sentence-transformers`)
- **Vector Database**: FAISS (`faiss-cpu`)
- **LLM SDK**: Hugging Face Hub (`huggingface_hub`)

---

## ⚙️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/Madhav189/DocuMind.git
   cd DocuMind
   ```

2. **Create a virtual environment (Optional but recommended)**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   - Create a file named `.env` in the root directory.
   - Add your Hugging Face API Token (Ensure it has "Make calls to the Serverless Inference API" enabled):
     ```env
     HUGGINGFACEHUB_API_TOKEN=your_token_here
     ```

---

## 🏃‍♂️ How to Run

Start the Streamlit application by running the following command in your terminal:

```bash
python -m streamlit run app.py
```

This will automatically open the application in your default web browser. From there, you can upload a PDF in the sidebar and start asking questions!
