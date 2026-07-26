import os
import ssl
import shutil
import warnings

# 1. Force HTTP libraries to use standard 'certifi' certificates
try:
    import certifi
    os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()
    os.environ["SSL_CERT_FILE"] = certifi.where()
except ImportError:
    pass

# 2. Stub out failing Windows certificate store calls gracefully
if hasattr(ssl, '_load_windows_store_certs'):
    ssl._load_windows_store_certs = lambda *args, **kwargs: []
elif hasattr(ssl.SSLContext, '_load_windows_store_certs'):
    ssl.SSLContext._load_windows_store_certs = lambda *args, **kwargs: []

from dotenv import load_dotenv
from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq

from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Load environment variables
load_dotenv()

# Configure dynamic working directories
working_dir = os.path.dirname(os.path.abspath(__file__))
db_directory = os.path.join(working_dir, "doc_vectorstore")

# Load lightweight embedding model to stay well within Streamlit Cloud 1 GB RAM limit
embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Initialize LLM via Groq
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0
)


def process_document_to_chroma_db(file_names):
    all_chunks = []
    
    for file_name in file_names:
        file_path = os.path.join(working_dir, file_name)
        
        # Parse PDF using UnstructuredPDFLoader
        loader = UnstructuredPDFLoader(
            file_path=file_path,
            strategy="hi_res"
        )
        
        documents = loader.load()

        # Split document into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=400
        )

        chunks = text_splitter.split_documents(documents)
        all_chunks.extend(chunks)

    # Initialize Chroma client and clear old collection cleanly without deleting SQLite system files
    vectordb = Chroma(
        persist_directory=db_directory,
        embedding_function=embedder
    )
    
    try:
        vectordb.delete_collection()
    except Exception:
        pass  # Safe fallback if collection doesn't exist yet

    # Store new document chunks in Chroma DB
    vectordb = Chroma.from_documents(
        documents=all_chunks,
        embedding=embedder,
        persist_directory=db_directory
    )
    return vectordb


def answer_question_with_history(user_question, chat_history):
    # Load existing Chroma database
    vectordb = Chroma(
        persist_directory=db_directory,
        embedding_function=embedder
    )

    # Setup retriever
    retriever = vectordb.as_retriever(
        search_kwargs={"k": 3}
    )

    # History-aware contextualization prompt
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history. Do NOT answer the question, "
        "just reformulate it if needed and otherwise return it as is."
    )

    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])

    history_aware_retriever = create_history_aware_retriever(
        llm, 
        retriever, 
        contextualize_q_prompt
    )

    # QA prompt template
    system_prompt = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer "
        "the question. If you don't know the answer, say that you "
        "don't know.\n\n"
        "{context}"
    )

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    qa_chain = create_stuff_documents_chain(llm, qa_prompt)

    # Combine into full RAG pipeline
    rag_chain = create_retrieval_chain(history_aware_retriever, qa_chain)

    response = rag_chain.invoke({
        "input": user_question,
        "chat_history": chat_history
    })

    return response["answer"]