import os
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import json
import pickle
import numpy as np
from dataclasses import dataclass

# PDF processing
import fitz  # PyMuPDF
import pdfplumber

# Text processing
from sentence_transformers import SentenceTransformer
import faiss
from langchain.text_splitter import RecursiveCharacterTextSplitter

# LLM integration
from llama_cpp import Llama
import requests
import zipfile
import hashlib

logger = logging.getLogger(__name__)

@dataclass
class Document:
    """Document class to store content with metadata"""
    content: str
    metadata: Dict[str, Any]
    page_number: Optional[int] = None
    source: Optional[str] = None

class ModelDownloader:
    """Handle model downloads from Hugging Face"""
    
    BASE_MODELS = {
        "mistral-7b-instruct-v0.1.Q4_K_M.gguf": {
            "url": "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q4_K_M.gguf",
            "size": "4.08 GB"
         }
        # "llama-2-7b-chat.Q4_K_M.gguf": {
        #     "url": "https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGUF/resolve/main/llama-2-7b-chat.Q4_K_M.gguf", 
        #     "size": "3.83 GB"
        # },
        # "codellama-7b-instruct.Q4_K_M.gguf": {
        #     "url": "https://huggingface.co/TheBloke/CodeLlama-7B-Instruct-GGUF/resolve/main/codellama-7b-instruct.Q4_K_M.gguf",
        #     "size": "3.83 GB"
        # }
    }
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
    
    def download_model(self, model_name: str) -> str:
        """Download model if not exists, return path"""
        if model_name not in self.BASE_MODELS:
            raise ValueError(f"Model {model_name} not supported")
        
        model_path = self.models_dir / model_name
        
        if model_path.exists():
            logger.info(f"Model {model_name} already exists")
            return str(model_path)
        
        logger.info(f"Downloading {model_name}...")
        url = self.BASE_MODELS[model_name]["url"]
        
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded_size = 0
            
            with open(model_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded_size += len(chunk)
                        
                        if total_size > 0:
                            progress = (downloaded_size / total_size) * 100
                            if downloaded_size % (1024 * 1024 * 100) == 0:  # Log every 100MB
                                logger.info(f"Downloaded {progress:.1f}%")
            
            logger.info(f"Successfully downloaded {model_name}")
            return str(model_path)
            
        except Exception as e:
            if model_path.exists():
                model_path.unlink()  # Remove partial download
            logger.error(f"Failed to download {model_name}: {str(e)}")
            raise

class PDFProcessor:
    """Handle PDF text extraction with page tracking using PyMuPDF and pdfplumber"""
    
    def __init__(self):
        pass
    
    def extract_text_with_pages(self, pdf_path: str) -> List[Document]:
        """Extract text from PDF with page information using PyMuPDF as primary, pdfplumber as fallback"""
        documents = []
        
        try:
            # Primary: PyMuPDF (faster and more reliable)
            doc = fitz.open(pdf_path)
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text = page.get_text()
                
                if text and text.strip():
                    document = Document(
                        content=text.strip(),
                        metadata={
                            'source': os.path.basename(pdf_path),
                            'page': page_num + 1,
                            'total_pages': len(doc),
                            'extractor': 'pymupdf'
                        },
                        page_number=page_num + 1,
                        source=pdf_path
                    )
                    documents.append(document)
            
            doc.close()
            logger.info(f"PyMuPDF extracted text from {len(documents)} pages in {pdf_path}")
                        
        except Exception as e:
            logger.warning(f"PyMuPDF failed for {pdf_path}, trying pdfplumber: {str(e)}")
            
            # Fallback to pdfplumber (better for complex layouts)
            try:
                with pdfplumber.open(pdf_path) as pdf:
                    for page_num, page in enumerate(pdf.pages, 1):
                        text = page.extract_text()
                        if text and text.strip():
                            document = Document(
                                content=text.strip(),
                                metadata={
                                    'source': os.path.basename(pdf_path),
                                    'page': page_num,
                                    'total_pages': len(pdf.pages),
                                    'extractor': 'pdfplumber'
                                },
                                page_number=page_num,
                                source=pdf_path
                            )
                            documents.append(document)
                            
                logger.info(f"pdfplumber extracted text from {len(documents)} pages in {pdf_path}")
                            
            except Exception as e2:
                logger.error(f"Both PDF extractors failed for {pdf_path}: {str(e2)}")
                raise Exception(f"Failed to extract text from {pdf_path}. Both PyMuPDF and pdfplumber failed.")
        
        if not documents:
            logger.warning(f"No text extracted from {pdf_path}")
            
        return documents
    
    def get_pdf_metadata(self, pdf_path: str) -> Dict[str, Any]:
        """Extract metadata from PDF using PyMuPDF"""
        try:
            doc = fitz.open(pdf_path)
            metadata = doc.metadata
            
            pdf_info = {
                'pages': len(doc),
                'title': metadata.get('title', ''),
                'author': metadata.get('author', ''),
                'subject': metadata.get('subject', ''),
                'creator': metadata.get('creator', ''),
                'producer': metadata.get('producer', ''),
                'creation_date': metadata.get('creationDate', ''),
                'modification_date': metadata.get('modDate', ''),
                'encrypted': doc.needs_pass,
                'file_size_bytes': os.path.getsize(pdf_path),
                'file_size_mb': round(os.path.getsize(pdf_path) / (1024 * 1024), 2)
            }
            
            doc.close()
            return pdf_info
            
        except Exception as e:
            logger.error(f"Failed to get metadata for {pdf_path}: {str(e)}")
            return {
                'pages': 0,
                'file_size_mb': round(os.path.getsize(pdf_path) / (1024 * 1024), 2) if os.path.exists(pdf_path) else 0,
                'error': str(e)
            }
    
    def extract_images(self, pdf_path: str, output_dir: str = None) -> List[str]:
        """Extract images from PDF (optional feature)"""
        if output_dir is None:
            return []
            
        try:
            doc = fitz.open(pdf_path)
            image_paths = []
            
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                image_list = page.get_images()
                
                for img_index, img in enumerate(image_list):
                    xref = img[0]
                    pix = fitz.Pixmap(doc, xref)
                    
                    if pix.n - pix.alpha < 4:  # GRAY or RGB
                        img_path = os.path.join(output_dir, f"page_{page_num+1}_img_{img_index+1}.png")
                        pix.save(img_path)
                        image_paths.append(img_path)
                    
                    pix = None
            
            doc.close()
            logger.info(f"Extracted {len(image_paths)} images from {pdf_path}")
            return image_paths
            
        except Exception as e:
            logger.error(f"Failed to extract images from {pdf_path}: {str(e)}")
            return []

class VectorStore:
    """FAISS-based vector store for document embeddings"""
    
    def __init__(self, embedding_model_name: str = "all-MiniLM-L6-v2"):
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.index = None
        self.documents = []
        self.embeddings = None
        self.dimension = self.embedding_model.get_sentence_embedding_dimension()
        
    def add_documents(self, documents: List[Document]):
        """Add documents to the vector store"""
        logger.info(f"Adding {len(documents)} documents to vector store...")
        
        # Extract text content
        texts = [doc.content for doc in documents]
        
        # Generate embeddings
        embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        
        # Initialize or update FAISS index
        if self.index is None:
            self.index = faiss.IndexFlatIP(self.dimension)  # Inner Product for similarity
            self.documents = documents
            self.embeddings = embeddings
        else:
            self.documents.extend(documents)
            self.embeddings = np.vstack([self.embeddings, embeddings])
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings.astype('float32'))
        
        logger.info(f"Vector store now contains {len(self.documents)} documents")
    
    def similarity_search(self, query: str, k: int = 3) -> List[Dict[str, Any]]:
        """Search for similar documents"""
        if self.index is None or len(self.documents) == 0:
            return []
        
        # Encode query
        query_embedding = self.embedding_model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # Search
        scores, indices = self.index.search(query_embedding.astype('float32'), k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.documents):
                doc = self.documents[idx]
                results.append({
                    'content': doc.content,
                    'metadata': doc.metadata,
                    'score': float(score),
                    'page': doc.page_number,
                    'source': doc.metadata.get('source', 'Unknown')
                })
        
        return results
    
    def save(self, path: str):
        """Save vector store to disk"""
        save_path = Path(path)
        save_path.mkdir(exist_ok=True)
        
        if self.index is not None:
            # Save FAISS index
            faiss.write_index(self.index, str(save_path / "index.faiss"))
            
            # Save documents and metadata
            with open(save_path / "documents.pkl", 'wb') as f:
                pickle.dump(self.documents, f)
            
            # Save embeddings
            np.save(save_path / "embeddings.npy", self.embeddings)
            
            logger.info(f"Vector store saved to {path}")
    
    def load(self, path: str) -> bool:
        """Load vector store from disk"""
        load_path = Path(path)
        
        try:
            if (load_path / "index.faiss").exists():
                # Load FAISS index
                self.index = faiss.read_index(str(load_path / "index.faiss"))
                
                # Load documents
                with open(load_path / "documents.pkl", 'rb') as f:
                    self.documents = pickle.load(f)
                
                # Load embeddings
                self.embeddings = np.load(load_path / "embeddings.npy")
                
                logger.info(f"Vector store loaded from {path}")
                return True
        except Exception as e:
            logger.error(f"Failed to load vector store: {str(e)}")
            return False
        
        return False

class RAGPipeline:
    """Main RAG Pipeline orchestrator"""
    
    def __init__(self, 
                 model_name: str = "mistral-7b-instruct-v0.1.Q4_K_M.gguf",
                 vector_store_path: str = "vector_store",
                 chunk_size: int = 1000,
                 chunk_overlap: int = 200,
                 max_tokens: int = 1000,
                 temperature: float = 0.1):
        
        self.model_name = model_name
        self.vector_store_path = vector_store_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        # Components
        self.model_downloader = ModelDownloader()
        self.pdf_processor = PDFProcessor()
        self.vector_store = VectorStore()
        self.llm = None
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
        )
        
        # Initialize
        self._initialize_llm()
        self._load_existing_vector_store()
    
    def _initialize_llm(self):
        """Initialize the language model"""
        try:
            logger.info("Initializing language model...")
            model_path = self.model_downloader.download_model(self.model_name)
            
            self.llm = Llama(
                model_path=model_path,
                n_ctx=4096,  # Context size
                n_batch=512,  # Batch size
                verbose=False,
                n_threads=os.cpu_count() - 1 if os.cpu_count() > 1 else 1
            )
            
            logger.info("Language model initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize LLM: {str(e)}")
            raise
    
    def _load_existing_vector_store(self):
        """Load existing vector store if available"""
        if os.path.exists(self.vector_store_path):
            if self.vector_store.load(self.vector_store_path):
                logger.info("Loaded existing vector store")
            else:
                logger.warning("Failed to load existing vector store")
    
    def process_documents(self, pdf_paths: List[str]) -> bool:
        """Process PDF documents and add to vector store"""
        try:
            all_documents = []
            
            for pdf_path in pdf_paths:
                logger.info(f"Processing {pdf_path}...")
                
                # Extract text with page info
                page_documents = self.pdf_processor.extract_text_with_pages(pdf_path)
                
                # Chunk documents
                chunked_documents = []
                for doc in page_documents:
                    chunks = self.text_splitter.split_text(doc.content)
                    
                    for i, chunk in enumerate(chunks):
                        if chunk.strip():  # Only add non-empty chunks
                            chunked_doc = Document(
                                content=chunk.strip(),
                                metadata={
                                    **doc.metadata,
                                    'chunk_id': i,
                                    'total_chunks': len(chunks)
                                },
                                page_number=doc.page_number,
                                source=doc.source
                            )
                            chunked_documents.append(chunked_doc)
                
                all_documents.extend(chunked_documents)
                logger.info(f"Created {len(chunked_documents)} chunks from {pdf_path}")
            
            # Add to vector store
            if all_documents:
                self.vector_store.add_documents(all_documents)
                self.vector_store.save(self.vector_store_path)
                logger.info(f"Successfully processed {len(pdf_paths)} PDF(s) with {len(all_documents)} total chunks")
                return True
            else:
                logger.warning("No documents to process")
                return False
                
        except Exception as e:
            logger.error(f"Error processing documents: {str(e)}")
            return False
    
    def _create_prompt(self, query: str, context_docs: List[Dict[str, Any]]) -> str:
        """Create prompt for the LLM"""
        
        context_text = ""
        for i, doc in enumerate(context_docs, 1):
            page_info = f" (Page {doc['page']})" if doc.get('page') else ""
            source_info = f" from {doc.get('source', 'Unknown')}" if doc.get('source') else ""
            context_text += f"\n[Context {i}]{page_info}{source_info}:\n{doc['content']}\n"
        
        prompt = f"""You are a helpful AI assistant that answers questions based on the provided context from PDF documents. 

Context from documents:
{context_text}

Question: {query}

Instructions:
1. Answer the question based on the provided context
2. Include relevant excerpts from the context in your response using quotes
3. Mention the page number when referencing specific information
4. If the context doesn't contain enough information to fully answer the question, say so
5. Be concise but thorough

Answer:"""
        
        return prompt
    
    def query(self, question: str, max_sources: int = 3) -> Dict[str, Any]:
        """Query the RAG system"""
        try:
            logger.info(f"Processing query: {question}")
            
            # Retrieve relevant documents
            relevant_docs = self.vector_store.similarity_search(question, k=max_sources)
            
            if not relevant_docs:
                return {
                    'answer': "I couldn't find any relevant information in the uploaded documents to answer your question.",
                    'sources': []
                }
            
            # Create prompt
            prompt = self._create_prompt(question, relevant_docs)
            
            # Generate response
            logger.info("Generating response...")
            response = self.llm(
                prompt,
                max_tokens=self.max_tokens,
                temperature=self.temperature,
                top_p=0.95,
                repeat_penalty=1.1,
                stop=["Question:", "\n\n"]
            )
            
            answer = response['choices'][0]['text'].strip()
            
            # Format sources
            sources = []
            for doc in relevant_docs:
                sources.append({
                    'content': doc['content'][:500] + "..." if len(doc['content']) > 500 else doc['content'],
                    'page': doc.get('page'),
                    'source': doc.get('source'),
                    'score': doc.get('score', 0.0)
                })
            
            logger.info("Response generated successfully")
            
            return {
                'answer': answer,
                'sources': sources,
                'query': question
            }
            
        except Exception as e:
            logger.error(f"Error during query: {str(e)}")
            return {
                'answer': f"I encountered an error while processing your question: {str(e)}",
                'sources': []
            }
    
    def is_ready(self) -> bool:
        """Check if the pipeline is ready for queries"""
        return (self.llm is not None and 
                self.vector_store is not None and 
                len(self.vector_store.documents) > 0)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pipeline statistics"""
        return {
            'model': self.model_name,
            'total_documents': len(self.vector_store.documents) if self.vector_store else 0,
            'chunk_size': self.chunk_size,
            'chunk_overlap': self.chunk_overlap,
            'is_ready': self.is_ready()
        }