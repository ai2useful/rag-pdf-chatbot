# RAG PDF Chatbot 📚🤖

A completely free, deployable RAG (Retrieval Augmented Generation) application that lets you chat with your PDF documents using open-source models. Built with Streamlit, it runs locally or in the cloud without requiring any paid APIs.

## ✨ Features

- **📄 Unlimited PDF Upload**: Upload single or multiple PDFs with no page limits
- **🔍 Smart Document Search**: Vector-based similarity search with FAISS
- **💬 Interactive Chat**: Multi-turn conversations with context awareness  
- **📖 Source Citations**: Every answer includes relevant excerpts with page numbers
- **🆓 Completely Free**: Uses only open-source models and tools
- **💻 Offline Capable**: Run entirely offline after initial model download
- **🚀 Easy Deployment**: Deploy locally, on cloud, or in containers
- **⚙️ Configurable**: Customize models, chunk sizes, and parameters

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **PDF Processing**: PyMuPDF (fitz), pdfplumber
- **Embeddings**: SentenceTransformers (all-MiniLM-L6-v2)
- **Vector Store**: FAISS (local storage)
- **LLM**: llama-cpp-python with GGUF models
- **Text Chunking**: LangChain

## 📋 Requirements

### System Requirements
- **OS**: Windows, macOS, or Linux
- **RAM**: 6+ GB (8GB+ recommended)
- **Storage**: 10+ GB free space (for models)
- **CPU**: Multi-core processor recommended

### Software Requirements
- **Python**: 3.9+
- **pip**: Latest version

## 🚀 Quick Start

### Option 1: Automatic Setup (Recommended)

1. **Clone the repository**:
```bash
git clone <repository-url>
cd rag-pdf-chatbot
```

2. **Run the setup script**:
```bash
python setup.py
```

3. **Start the application**:
```bash
streamlit run app.py
```

4. **Open your browser** to `http://localhost:8501`

### Option 2: Manual Setup

1. **Clone and navigate**:
```bash
git clone <repository-url>
cd rag-pdf-chatbot
```

2. **Create virtual environment** (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Create directories**:
```bash
mkdir logs uploaded_pdfs vector_store models backups
```

5. **Run the app**:
```bash
streamlit run app.py
```

## 🐳 Docker Deployment

### Build and Run with Docker

```bash
# Build the image
docker build -t rag-pdf-chatbot .

# Run the container
docker run -p 8501:8501 -v $(pwd)/models:/app/models rag-pdf-chatbot
```

### Using Docker Compose

```yaml
version: '3.8'
services:
  rag-chatbot:
    build: .
    ports:
      - "8501:8501"
    volumes:
      - ./models:/app/models
      - ./uploaded_pdfs:/app/uploaded_pdfs
      - ./vector_store:/app/vector_store
```

## 📖 How to Use

### 1. Upload PDFs
- Click "Choose PDF files" in the sidebar
- Select one or more PDF documents
- Click "🚀 Process PDFs"

### 2. Wait for Processing  
- The app will extract text, create chunks, and build embeddings
- Processing time depends on document size
- You'll see "✅ Ready to chat!" when complete

### 3. Start Chatting
- Type your question in the chat input
- Get answers with source citations
- Ask follow-up questions for clarification

### 4. Review Sources
- Each answer includes source excerpts
- Page numbers help you locate original content
- Expand "📖 Sources" to see more context

## ⚙️ Configuration

### Model Selection
Choose from these free models:
- **Mistral 7B Instruct** (Default): Fast, general-purpose
- **Llama 2 7B Chat**: Conversational AI
- **Code Llama 7B**: Best for technical documents

### Advanced Settings
- **Chunk Size**: Text chunk size (200-2000 characters)
- **Chunk Overlap**: Overlap between chunks (0-500 characters) 
- **Max Sources**: Number of sources to retrieve (1-10)

## 📁 Project Structure

```
rag-pdf-chatbot/
├── app.py                 # Main Streamlit application
├── rag_pipeline.py        # Core RAG pipeline logic
├── utils.py              # Utility functions
├── requirements.txt      # Python dependencies
├── setup.py             # Automated setup script
├── Dockerfile           # Container configuration
├── README.md           # This file
├── config.json         # Configuration settings
└── directories/
    ├── logs/           # Application logs
    ├── models/         # Downloaded LLM models
    ├── uploaded_pdfs/  # Temporary PDF storage
    ├── vector_store/   # FAISS embeddings
    └── backups/        # Backup files
```

## 🎯 Model Information

### Available Models

| Model | Size | Best For | Memory |
|-------|------|----------|---------|
| Mistral 7B Instruct | ~4.1 GB | General Q&A | 6+ GB RAM |
| Llama 2 7B Chat | ~3.8 GB | Conversations | 6+ GB RAM |
| Code Llama 7B | ~3.8 GB | Technical docs | 6+ GB RAM |

### Model Downloads
- Models download automatically on first use
- Downloaded to `models/` directory
- Can be shared across sessions
- Offline usage after download

## 💡 Tips for Best Results

### Document Preparation
- Use text-based PDFs (not scanned images)
- Ensure good text quality and formatting
- Smaller documents process faster

### Asking Questions
- Be specific and clear
- Use keywords from your documents
- Ask follow-up questions for clarity
- Check source citations for accuracy

### Performance Optimization
- Close other applications for more RAM
- Use smaller chunk sizes for better precision
- Reduce max sources if responses are slow

## 🔧 Troubleshooting

### Common Issues

**"Model failed to download"**
- Check internet connection
- Ensure sufficient disk space
- Try restarting the application

**"PDF processing failed"**
- Verify PDF is not corrupted
- Try with a smaller PDF first
- Check file permissions

**"Out of memory errors"**
- Close other applications
- Use a smaller model
- Reduce chunk size and batch size

**"Slow responses"**
- Check system resources
- Reduce max sources
- Use a faster model

### Performance Issues
- **High RAM usage**: Normal for large models
- **Slow startup**: First-time model download
- **Long processing**: Large PDFs take time

## 🌐 Deployment Options

### Local Development
```bash
streamlit run app.py
```

### Production Deployment
```bash
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

### Cloud Deployment
- **Hugging Face Spaces**: Free hosting option
- **Streamlit Cloud**: Direct GitHub integration
- **Railway/Render**: Container-based deployment
- **AWS/GCP/Azure**: Full cloud deployment

### Environment Variables
```bash
export STREAMLIT_SERVER_PORT=8501
export STREAMLIT_SERVER_ADDRESS=0.0.0.0
export STREAMLIT_BROWSER_GATHER_USAGE_STATS=false
export STREAMLIT_SERVER_HEADLESS=true
```

## 🔒 Privacy & Security

### Data Privacy
- **All processing happens locally** - your documents never leave your machine
- **No external API calls** for document processing or chat (except model downloads)
- **No data collection** or tracking
- **Complete offline capability** after initial setup

### Security Features
- Documents are processed locally only
- No sensitive data transmitted to external servers
- Models run entirely on your hardware
- Automatic cleanup of temporary files

## 📊 Monitoring & Logs

### Application Logs
- Located in `logs/rag_chatbot.log`
- Configurable log levels (DEBUG, INFO, WARNING, ERROR)
- Automatic log rotation

### System Monitoring
- Built-in system resource monitoring
- Memory usage tracking
- Processing time metrics
- Model performance stats

## 🧪 Testing

### Run Tests
```bash
# Install test dependencies
pip install pytest pytest-cov

# Run tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=. --cov-report=html
```

### Manual Testing
1. Upload a small PDF (1-5 pages)
2. Ask simple questions about the content
3. Verify sources and citations
4. Test with different question types

## 🚀 Advanced Usage

### Custom Models
Add your own GGUF models to the `models/` directory and update the model list in `rag_pipeline.py`:

```python
BASE_MODELS = {
    "your-custom-model.gguf": {
        "url": "https://your-model-url.com/model.gguf",
        "size": "X.X GB"
    }
}
```

### Configuration File
Create `config.json` for persistent settings:

```json
{
    "default_model": "mistral-7b-instruct-v0.1.Q4_K_M.gguf",
    "chunk_size": 1000,
    "chunk_overlap": 200,
    "max_sources": 3,
    "max_tokens": 1000,
    "temperature": 0.1,
    "embedding_model": "all-MiniLM-L6-v2"
}
```

### Batch Processing
Process multiple PDFs programmatically:

```python
from rag_pipeline import RAGPipeline

# Initialize pipeline
rag = RAGPipeline()

# Process multiple files
pdf_files = ["doc1.pdf", "doc2.pdf", "doc3.pdf"]
rag.process_documents(pdf_files)

# Query the system
response = rag.query("What are the main topics discussed?")
print(response['answer'])
```

## 🔄 Updates & Maintenance

### Updating the Application
```bash
# Pull latest changes
git pull origin main

# Update dependencies
pip install -r requirements.txt --upgrade

# Restart the application
streamlit run app.py
```

### Model Updates
- New models can be added to the `BASE_MODELS` dictionary
- Old models can be removed from the `models/` directory
- Vector stores are model-agnostic and don't need rebuilding

### Database Maintenance
```bash
# Clear vector store
rm -rf vector_store/*

# Clear uploaded files
rm -rf uploaded_pdfs/*

# Clear logs
rm -rf logs/*
```

## 🤝 Contributing

### Development Setup
```bash
# Clone repository
git clone <repository-url>
cd rag-pdf-chatbot

# Create development environment
python -m venv dev-env
source dev-env/bin/activate

# Install development dependencies
pip install -r requirements.txt
pip install pytest black flake8 pre-commit

# Install pre-commit hooks
pre-commit install
```

### Code Standards
- **Formatting**: Black code formatter
- **Linting**: Flake8
- **Type Hints**: Encouraged for new code
- **Documentation**: Docstrings for all functions
- **Testing**: Unit tests for core functionality

### Submitting Changes
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Run tests and linting
6. Submit a pull request

## 📚 Additional Resources

### Documentation
- [Streamlit Documentation](https://docs.streamlit.io/)
- [LangChain Documentation](https://docs.langchain.com/)
- [FAISS Documentation](https://faiss.ai/)
- [llama.cpp Documentation](https://github.com/ggerganov/llama.cpp)

### Model Sources
- [Hugging Face Model Hub](https://huggingface.co/models)
- [TheBloke's GGUF Models](https://huggingface.co/TheBloke)

### Community
- [Streamlit Community](https://discuss.streamlit.io/)
- [LangChain Discord](https://discord.gg/langchain)
- [r/LocalLLaMA](https://reddit.com/r/LocalLLaMA)

## ❓ FAQ

**Q: How much storage do I need?**
A: At minimum 10GB. Each model is 3-4GB, plus space for documents and vector stores.

**Q: Can I use this commercially?**
A: Check individual model licenses. Most are free for commercial use, but verify before deployment.

**Q: How accurate are the responses?**
A: Accuracy depends on document quality and question specificity. Always verify important information.

**Q: Can I add more file types?**
A: Currently supports PDF only. Adding Word, TXT, etc. is planned for future versions.

**Q: Does this work with scanned PDFs?**
A: Not directly. You'll need OCR preprocessing for image-based PDFs.

**Q: How do I improve response quality?**
A: Use high-quality documents, ask specific questions, and check source citations.

## 🐛 Known Issues

- Large PDFs (>500 pages) may take significant processing time
- Memory usage can be high with large models
- First-time model downloads require internet connection
- Some PDFs with complex formatting may not extract perfectly

## 🗺️ Roadmap

### Upcoming Features
- [ ] Support for Word documents, TXT files
- [ ] OCR integration for scanned PDFs  
- [ ] Document summarization
- [ ] Multi-language support
- [ ] API endpoint for programmatic access
- [ ] Web interface improvements
- [ ] Mobile-responsive design
- [ ] Export Q&A history

### Performance Improvements
- [ ] Streaming responses
- [ ] Background processing
- [ ] Model quantization options
- [ ] GPU acceleration support
- [ ] Caching optimizations

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Streamlit**: For the amazing web framework
- **Hugging Face**: For model hosting and transformers
- **TheBloke**: For providing quantized GGUF models
- **LangChain**: For text processing utilities
- **Meta AI**: For Llama models
- **Mistral AI**: For Mistral models
- **Facebook Research**: For FAISS vector search

## 📞 Support

If you encounter issues or have questions:

1. Check the [FAQ](#-faq) section
2. Review [troubleshooting](#-troubleshooting) guide
3. Search existing [GitHub issues](link-to-issues)
4. Create a new issue with detailed description

---

**Made with ❤️ using free and open-source tools**

*Happy chatting with your PDFs! 📚✨*