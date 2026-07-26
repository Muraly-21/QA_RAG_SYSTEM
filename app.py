import os
import pypdf
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from rag_utility import process_document_to_chroma_db, answer_question_with_history

# Set the working directory
working_dir = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(page_title="Multi-Document Conversational AI", page_icon="📚")
st.title("📚 Multi-Document Conversational AI")

# Resource limits to prevent Streamlit Cloud 1 GB RAM crashes (e.g. 300-page uploads)
MAX_FILE_SIZE_MB = 15
MAX_PAGES_PER_FILE = 40

# Memory management vaults to maintain state across reruns
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "display_history" not in st.session_state:
    st.session_state.display_history = []

# File uploader widget
uploaded_files = st.file_uploader("Upload PDF Files", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    current_files_signature = [file.name for file in uploaded_files]

    # Process only if the set of uploaded files has changed
    if "processed_files" not in st.session_state or st.session_state.processed_files != current_files_signature:
        
        #Safety guardrail for Memory & Large Files
        invalid_file = False
        for file in uploaded_files:
            # 1. File size check
            size_mb = file.size / (1024 * 1024)
            if size_mb > MAX_FILE_SIZE_MB:
                st.error(f"⚠️ '{file.name}' is too large ({size_mb:.1f} MB). Please keep uploads under {MAX_FILE_SIZE_MB} MB for the free cloud demo.")
                invalid_file = True
                break

            # 2. Page count check
            try:
                reader = pypdf.PdfReader(file)
                page_count = len(reader.pages)
                if page_count > MAX_PAGES_PER_FILE:
                    st.error(f"⚠️ '{file.name}' has {page_count} pages. To prevent server memory crashes on the free tier, please upload files with under {MAX_PAGES_PER_FILE} pages.")
                    invalid_file = True
                    break
            except Exception as e:
                st.error(f"Could not read PDF structure for '{file.name}': {e}")
                invalid_file = True
                break

        # Stop processing if any file violates limits
        if invalid_file:
            st.stop()

        # Save files to disk temporarily for parsing
        saved_file_names = []
        for uploaded_file in uploaded_files:
            save_path = os.path.join(working_dir, uploaded_file.name)
            with open(save_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            saved_file_names.append(uploaded_file.name)

        # Process document through vector DB
        with st.spinner("Analyzing all Documents..."):
            process_document_to_chroma_db(saved_file_names)

        # Clean up temporary PDF files from disk after embedding
        for file_name in saved_file_names:
            file_path = os.path.join(working_dir, file_name)
            if os.path.exists(file_path):
                os.remove(file_path)

        # Reset session chat history for fresh files
        st.session_state.processed_files = current_files_signature
        st.session_state.chat_history = []
        st.session_state.display_history = []
        st.success("All Documents Processed Successfully!")

# Display stored display history
for message in st.session_state.display_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Interactive chat input bar
if user_query := st.chat_input("Ask something from the uploaded documents..."):
    if not uploaded_files:
        st.warning("Please upload at least one PDF file first.")
    else:
        # Display and record user prompt
        with st.chat_message("user"):
            st.markdown(user_query)
        st.session_state.display_history.append({"role": "user", "content": user_query})

        # Process query through RAG pipeline
        with st.spinner("Thinking..."):
            bot_response = answer_question_with_history(user_query, st.session_state.chat_history)

        # Display assistant response
        with st.chat_message("assistant"):
            st.markdown(bot_response)

        # Record assistant response
        st.session_state.display_history.append({"role": "assistant", "content": bot_response})

        # Append messages using LangChain format for history context
        st.session_state.chat_history.extend([
            HumanMessage(content=user_query),
            AIMessage(content=bot_response)
        ])