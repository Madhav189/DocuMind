import streamlit as st
import tempfile
import os
from utils import process_pdf, create_vector_store, get_answer

st.set_page_config(page_title="DocuMind", page_icon="🧠")

st.title("DocuMind - Chat with your PDF")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize vector store
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

with st.sidebar:
    st.header("Upload Document")
    uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    
    if uploaded_file is not None and st.session_state.vector_store is None:
        with st.spinner("Processing PDF..."):
            # Save temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_path = tmp_file.name
                
            try:
                # Process and create vector store
                chunks = process_pdf(tmp_path)
                st.session_state.vector_store = create_vector_store(chunks)
                st.success("PDF processed and indexed successfully!")
            except Exception as e:
                st.error(f"Error processing PDF: {e}")
            finally:
                # Clean up temp file
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# React to user input
if prompt := st.chat_input("Ask a question about your document"):
    if st.session_state.vector_store is None:
        st.warning("Please upload a PDF first.")
    else:
        # Display user message in chat message container
        st.chat_message("user").markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.spinner("Thinking..."):
            # Get answer
            response = get_answer(prompt, st.session_state.vector_store)
            
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(response)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
