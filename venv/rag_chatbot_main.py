import streamlit as st
import os
import tempfile
import shutil
from pathlib import Path
import logging
from typing import List, Dict, Any, Optional
import time
import json

# Import our custom modules
from rag_pipeline import RAGPipeline
from utils import setup_logging, validate_pdf, get_pdf_metadata, estimate_processing_time

# Configure logging
setup_logging()
logger = logging.getLogger(__name__)

# Page config
st.set_page_config(
    page_title="RAG PDF Chatbot",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
UPLOAD_DIR = "uploaded_pdfs"
VECTOR_STORE_DIR = "vector_store"
MODELS_DIR = "models"

def initialize_directories():
    """Create necessary directories if they don't exist"""
    for dir_path in [UPLOAD_DIR, VECTOR_STORE_DIR, MODELS_DIR]:
        Path(dir_path).mkdir(exist_ok=True)

def initialize_session_state():
    """Initialize session state variables"""
    if 'rag_pipeline' not in st.session_state:
        st.session_state.rag_pipeline = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'processed_files' not in st.session_state:
        st.session_state.processed_files = []
    if 'processing_status' not in st.session_state:
        st.session_state.processing_status = ""

def save_uploaded_file(uploaded_file) -> str:
    """Save uploaded file to temp directory and return path"""
    file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

def process_pdfs(pdf_files: List[Any], model_choice: str, chunk_size: int, chunk_overlap: int):
    """Process uploaded PDFs through the RAG pipeline"""
    try:
        # Initialize RAG pipeline
        with st.spinner("Initializing RAG pipeline..."):
            st.session_state.rag_pipeline = RAGPipeline(
                model_name=model_choice,
                vector_store_path=VECTOR_STORE_DIR,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
        
        # Process each PDF
        all_file_paths = []
        pdf_info = []
        
        for uploaded_file in pdf_files:
            with st.spinner(f"Saving {uploaded_file.name}..."):
                file_path = save_uploaded_file(uploaded_file)
                
                # Validate PDF
                if not validate_pdf(file_path):
                    st.error(f"❌ Invalid PDF file: {uploaded_file.name}")
                    continue
                
                # Get PDF metadata for better user feedback
                metadata = get_pdf_metadata(file_path)
                pdf_info.append({
                    'name': uploaded_file.name,
                    'path': file_path,
                    'pages': metadata.get('pages', 0),
                    'size_mb': metadata.get('file_size_mb', 0),
                    'text_extractable': metadata.get('text_extractable', True),
                    'estimated_time': metadata.get('estimated_processing_time', 'Unknown')
                })
                    
                all_file_paths.append(file_path)
        
        if not all_file_paths:
            st.error("❌ No valid PDF files to process")
            return False
        
        # Display processing info
        st.info(f"📊 Processing {len(all_file_paths)} PDF(s):")
        for info in pdf_info:
            cols = st.columns([3, 1, 1, 1])
            with cols[0]:
                st.write(f"📄 {info['name']}")
            with cols[1]:
                st.write(f"{info['pages']} pages")
            with cols[2]:
                st.write(f"{info['size_mb']:.1f} MB")
            with cols[3]:
                if info['text_extractable']:
                    st.write("✅ Text OK")
                else:
                    st.write("⚠️ Scanned PDF")
        
        # Show total estimated time
        total_pages = sum(info['pages'] for info in pdf_info)
        st.info(f"📈 Total: {total_pages} pages • Estimated processing time: {estimate_processing_time(total_pages, sum(info['size_mb'] for info in pdf_info))}")
        
        # Process all PDFs
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        with st.spinner("Processing PDFs and building vector store..."):
            status_text.text("🔍 Extracting text from PDFs...")
            progress_bar.progress(0.3)
            
            status_text.text("🧩 Creating text chunks...")
            progress_bar.progress(0.6)
            
            status_text.text("🔢 Generating embeddings...")
            progress_bar.progress(0.8)
            
            success = st.session_state.rag_pipeline.process_documents(all_file_paths)
            progress_bar.progress(1.0)
            
            if success:
                status_text.text("✅ Processing completed!")
                st.session_state.processed_files = [info['name'] for info in pdf_info]
                
                # Show processing summary
                st.success(f"🎉 Successfully processed {len(all_file_paths)} PDF(s) with {total_pages} total pages!")
                
                # Display extraction method used
                with st.expander("📋 Processing Details", expanded=False):
                    stats = st.session_state.rag_pipeline.get_stats()
                    st.write(f"**Total documents processed:** {stats.get('total_documents', 0)}")
                    st.write(f"**Chunk size:** {stats.get('chunk_size', 0)} characters")
                    st.write(f"**Chunk overlap:** {stats.get('chunk_overlap', 0)} characters")
                    st.write(f"**Model:** {stats.get('model', 'Unknown')}")
                
                return True
            else:
                status_text.text("❌ Processing failed!")
                st.error("❌ Failed to process PDFs")
                return False
            
    except Exception as e:
        logger.error(f"Error processing PDFs: {str(e)}")
        st.error(f"❌ Error: {str(e)}")
        return False

def display_chat_interface():
    """Display the chat interface"""
    st.subheader("💬 Chat with your PDFs")
    
    # Display processed files
    if st.session_state.processed_files:
        with st.expander("📑 Processed Files", expanded=False):
            for file in st.session_state.processed_files:
                st.write(f"• {file}")
    
    # Chat history
    chat_container = st.container()
    with chat_container:
        for i, (user_msg, bot_msg) in enumerate(st.session_state.chat_history):
            # User message
            with st.chat_message("user"):
                st.write(user_msg)
            
            # Bot message
            with st.chat_message("assistant"):
                st.write(bot_msg['answer'])
                
                # Display sources if available
                if bot_msg.get('sources'):
                    with st.expander("📖 Sources", expanded=False):
                        for j, source in enumerate(bot_msg['sources']):
                            st.markdown(f"**Source {j+1}:**")
                            st.markdown(f"> {source['content']}")
                            if source.get('page'):
                                st.caption(f"Page: {source['page']}")
                            st.divider()
    
    # Chat input
    if st.session_state.rag_pipeline and st.session_state.rag_pipeline.is_ready():
        user_question = st.chat_input("Ask a question about your PDFs...")
        
        if user_question:
            # Add user message to history
            with st.chat_message("user"):
                st.write(user_question)
            
            # Get bot response
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    try:
                        response = st.session_state.rag_pipeline.query(user_question)
                        
                        # Display answer
                        st.write(response['answer'])
                        
                        # Display sources
                        if response.get('sources'):
                            with st.expander("📖 Sources", expanded=True):
                                for j, source in enumerate(response['sources']):
                                    st.markdown(f"**Source {j+1}:**")
                                    st.markdown(f"> {source['content']}")
                                    if source.get('page'):
                                        st.caption(f"Page: {source['page']}")
                                    st.divider()
                        
                        # Add to chat history
                        st.session_state.chat_history.append((user_question, response))
                        
                    except Exception as e:
                        error_msg = f"❌ Error generating response: {str(e)}"
                        st.error(error_msg)
                        logger.error(f"Query error: {str(e)}")
    else:
        st.info("👆 Please upload and process PDFs first to start chatting!")

def main():
    """Main application function"""
    initialize_directories()
    initialize_session_state()
    
    # Header
    st.title("📚 RAG PDF Chatbot")
    st.markdown("Upload PDFs and chat with them using AI! Completely free and runs locally.")
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model selection
        model_choice = st.selectbox(
            "🤖 Choose Model",
            options=[
                "mistral-7b-instruct-v0.1.Q4_K_M.gguf"
                # "llama-2-7b-chat.Q4_K_M.gguf",
                # "codellama-7b-instruct.Q4_K_M.gguf"
            ],
            index=0,
            help="Select the language model for generating responses"
        )
        
        # Advanced settings
        with st.expander("🔧 Advanced Settings", expanded=False):
            chunk_size = st.slider("Chunk Size", 200, 2000, 1000, 50, 
                                 help="Size of text chunks for processing")
            chunk_overlap = st.slider("Chunk Overlap", 0, 500, 200, 25,
                                    help="Overlap between consecutive chunks")
            max_sources = st.slider("Max Sources", 1, 10, 3,
                                  help="Maximum number of sources to retrieve")
        
        # Clear conversation
        if st.button("🗑️ Clear Conversation", type="secondary"):
            st.session_state.chat_history = []
            st.rerun()
        
        # Reset everything
        if st.button("🔄 Reset All", type="secondary"):
            st.session_state.chat_history = []
            st.session_state.processed_files = []
            st.session_state.rag_pipeline = None
            # Clear uploaded files
            if os.path.exists(UPLOAD_DIR):
                shutil.rmtree(UPLOAD_DIR)
                os.makedirs(UPLOAD_DIR)
            st.rerun()
    
    # Main content area
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("📄 Upload PDFs")
        
        uploaded_files = st.file_uploader(
            "Choose PDF files",
            type="pdf",
            accept_multiple_files=True,
            help="Upload one or more PDF files to chat with"
        )
        
        if uploaded_files:
            st.write(f"📁 {len(uploaded_files)} file(s) selected:")
            for file in uploaded_files:
                st.write(f"• {file.name} ({file.size / 1024 / 1024:.1f} MB)")
            
            if st.button("🚀 Process PDFs", type="primary"):
                success = process_pdfs(uploaded_files, model_choice, chunk_size, chunk_overlap)
                if success:
                    st.rerun()
        
        # System status
        st.subheader("📊 System Status")
        if st.session_state.rag_pipeline:
            if st.session_state.rag_pipeline.is_ready():
                st.success("✅ Ready to chat!")
            else:
                st.warning("⚠️ Pipeline not fully initialized")
        else:
            st.info("ℹ️ No PDFs processed yet")
        
        # Display processed files count
        if st.session_state.processed_files:
            st.metric("Processed Files", len(st.session_state.processed_files))
        
        # Model info
        if st.session_state.rag_pipeline:
            with st.expander("🤖 Model Info", expanded=False):
                st.write(f"**Model:** {model_choice}")
                st.write(f"**Chunk Size:** {chunk_size}")
                st.write(f"**Chunk Overlap:** {chunk_overlap}")
    
    with col2:
        display_chat_interface()
    
    # Footer
    st.markdown("---")
    st.markdown(
        "💡 **Tips:** "
        "• Ask specific questions for better results "
        "• Use keywords from your documents "
        "• Check the sources to verify information"
    )

if __name__ == "__main__":
    main()