#!/usr/bin/env python3
"""
Setup script for RAG PDF Chatbot
This script handles the initial setup and installation of dependencies.
"""

import os
import sys
import subprocess
import platform
import logging
from pathlib import Path

def setup_logging():
    """Setup logging for the setup process"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 9):
        logging.error("Python 3.9+ is required. Current version: %s.%s", version.major, version.minor)
        return False
    
    logging.info("Python version %s.%s.%s - Compatible ✓", version.major, version.minor, version.micro)
    return True

def check_system_requirements():
    """Check system requirements"""
    import psutil
    
    # Check RAM
    memory = psutil.virtual_memory()
    memory_gb = memory.total / (1024**3)
    
    if memory_gb < 4:
        logging.warning("⚠️ Less than 4GB RAM detected (%.1fGB). Performance may be limited.", memory_gb)
    else:
        logging.info("Memory: %.1fGB - Sufficient ✓", memory_gb)
    
    # Check disk space
    disk = psutil.disk_usage('/')
    free_gb = disk.free / (1024**3)
    
    if free_gb < 10:
        logging.warning("⚠️ Less than 10GB free disk space (%.1fGB). Models require significant storage.", free_gb)
    else:
        logging.info("Disk space: %.1fGB free - Sufficient ✓", free_gb)
    
    # Check CPU cores
    cpu_count = os.cpu_count()
    if cpu_count < 2:
        logging.warning("⚠️ Less than 2 CPU cores detected (%d). Performance may be slow.", cpu_count)
    else:
        logging.info("CPU cores: %d - Sufficient ✓", cpu_count)

def install_requirements():
    """Install Python requirements"""
    try:
        logging.info("Installing Python requirements...")
        
        # Upgrade pip first
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
        
        # Install requirements
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        
        logging.info("Requirements installed successfully ✓")
        return True
        
    except subprocess.CalledProcessError as e:
        logging.error("Failed to install requirements: %s", e)
        return False

def create_directories():
    """Create necessary directories"""
    dirs = [
        "logs",
        "uploaded_pdfs", 
        "vector_store",
        "models",
        "backups"
    ]
    
    for dir_name in dirs:
        Path(dir_name).mkdir(exist_ok=True)
        logging.info("Created directory: %s", dir_name)

def download_small_model():
    """Download a small model for testing"""
    try:
        logging.info("Downloading sentence transformer model for embeddings...")
        
        # This will download the embedding model on first use
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        logging.info("Embedding model downloaded successfully ✓")
        return True
        
    except Exception as e:
        logging.error("Failed to download embedding model: %s", e)
        return False

def create_config_file():
    """Create a default configuration file"""
    config = {
        "default_model": "mistral-7b-instruct-v0.1.Q4_K_M.gguf",
        "chunk_size": 1000,
        "chunk_overlap": 200,
        "max_sources": 3,
        "max_tokens": 1000,
        "temperature": 0.1
    }
    
    try:
        import json
        with open("config.json", "w") as f:
            json.dump(config, f, indent=2)
        
        logging.info("Created default config.json ✓")
        return True
        
    except Exception as e:
        logging.error("Failed to create config file: %s", e)
        return False

def test_installation():
    """Test if the installation works"""
    try:
        logging.info("Testing installation...")
        
        # Test imports
        import streamlit
        import fitz  # PyMuPDF
        import sentence_transformers
        import faiss
        import llama_cpp
        
        logging.info("All imports successful ✓")
        
        # Test sentence transformer
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer('all-MiniLM-L6-v2')
        test_embedding = model.encode(["This is a test sentence"])
        
        if test_embedding is not None and len(test_embedding) > 0:
            logging.info("Embedding model test successful ✓")
        
        logging.info("Installation test completed successfully ✓")
        return True
        
    except Exception as e:
        logging.error("Installation test failed: %s", e)
        return False

def print_next_steps():
    """Print next steps for the user"""
    print("\n" + "="*60)
    print("🎉 RAG PDF Chatbot Setup Complete!")
    print("="*60)
    print("\nNext steps:")
    print("1. Run the application:")
    print("   streamlit run app.py")
    print("\n2. Open your browser to:")
    print("   http://localhost:8501")
    print("\n3. Upload your PDF files and start chatting!")
    print("\nNotes:")
    print("• The first time you use a model, it will be downloaded automatically")
    print("• Large models (3-4GB) may take time to download")
    print("• For offline use, ensure models are downloaded beforehand")
    print("\nFor help and documentation, see README.md")
    print("="*60)

def main():
    """Main setup function"""
    setup_logging()
    
    print("🚀 RAG PDF Chatbot Setup")
    print("=" * 40)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Check system requirements
    check_system_requirements()
    
    # Create directories
    create_directories()
    
    # Install requirements
    if not install_requirements():
        logging.error("Setup failed during requirements installation")
        sys.exit(1)
    
    # Download small model for testing
    if not download_small_model():
        logging.warning("Could not download embedding model. It will be downloaded on first use.")
    
    # Create config file
    create_config_file()
    
    # Test installation
    if not test_installation():
        logging.error("Setup failed during testing")
        sys.exit(1)
    
    # Print next steps
    print_next_steps()

if __name__ == "__main__":
    main()