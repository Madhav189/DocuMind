import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

def process_pdf(file_path):
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = text_splitter.split_documents(docs)
    return chunks

def create_vector_store(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local("faiss_index")
    return vector_store

def get_answer(query, vector_store):
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(query)
    context = "\n\n".join([doc.page_content for doc in docs])
    
    hf_api_key = os.getenv("HUGGINGFACEHUB_API_TOKEN")
    
    if hf_api_key:
        try:
            import requests
            headers = {
                "Authorization": f"Bearer {hf_api_key}",
                "Content-Type": "application/json"
            }
            messages = [
                {"role": "system", "content": "You are a helpful AI assistant. Use the provided context to answer the user's question. If you don't know the answer, say you don't know."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"}
            ]
            payload = {
                "model": "microsoft/Phi-3-mini-4k-instruct",
                "messages": messages,
                "temperature": 0.1,
                "max_tokens": 512
            }
            response = requests.post(
                "https://router.huggingface.co/hf-inference/models/microsoft/Phi-3-mini-4k-instruct/v1/chat/completions",
                headers=headers,
                json=payload
            )
            
            if response.status_code == 200:
                data = response.json()
                return data["choices"][0]["message"]["content"]
            else:
                return f"LLM API Error: {response.status_code} - {response.text}\n\nFallback context:\n{context}"
        except Exception as e:
            return f"LLM Error: {str(e)}\n\nFallback context:\n{context}"
    else:
        return f"No LLM configured. Raw context retrieved:\n\n{context}"
