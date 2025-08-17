import os
import logging
import sys
from pathlib import Path
import fitz  # PyMuPDF
import tempfile
import shutil
from typing import Optional, List, Dict, Any
import streamlit as st

def setup_logging(log_level: str = "INFO") -> None:
    """Setup logging configuration"""
    
    # Create logs directory
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(logs_dir / "rag_chatbot.log"),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Reduce noise from other libraries
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

def validate_pdf(file_path: str) -> bool:
    """Validate if file is a proper PDF using PyMuPDF"""
    try:
        if not os.path.exists(file_path):
            return False
        
        if not file_path.lower().endswith('.pdf'):
            return False
        
        # Try to open with PyMuPDF to validate
        doc = fitz.open(file_path)
        
        # Check if we can read at least one page
        if len(doc) > 0:
            # Try to extract text from first page
            first_page = doc.load_page(0)
            first_page.get_text()
            doc.close()
            return True
        
        doc.close()
        return False
        
    except Exception as e:
        logging.getLogger(__name__).warning(f"PDF validation failed for {file_path}: {str(e)}")
        return False

def get_file_size_mb(file_path: str) -> float:
    """Get file size in MB"""
    try:
        size_bytes = os.path.getsize(file_path)
        size_mb = size_bytes / (1024 * 1024)
        return round(size_mb, 2)
    except:
        return 0.0

def cleanup_temp_files(directory: str) -> None:
    """Clean up temporary files in directory"""
    try:
        if os.path.exists(directory):
            shutil.rmtree(directory)
            os.makedirs(directory, exist_ok=True)
            logging.getLogger(__name__).info(f"Cleaned up directory: {directory}")
    except Exception as e:
        logging.getLogger(__name__).error(f"Failed to cleanup {directory}: {str(e)}")

def format_file_size(size_bytes: int) -> str:
    """Format file size in human readable format"""
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    import math
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return f"{s} {size_names[i]}"

def check_system_requirements() -> Dict[str, Any]:
    """Check system requirements and resources"""
    import psutil
    
    # Get system info
    cpu_count = os.cpu_count()
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    
    requirements = {
        "cpu_cores": cpu_count,
        "total_memory_gb": round(memory.total / (1024**3), 2),
        "available_memory_gb": round(memory.available / (1024**3), 2),
        "memory_percent": memory.percent,
        "total_disk_gb": round(disk.total / (1024**3), 2),
        "free_disk_gb": round(disk.free / (1024**3), 2),
        "disk_percent": round((disk.used / disk.total) * 100, 1)
    }
    
    # Check if system meets minimum requirements
    warnings = []
    if requirements["cpu_cores"] < 2:
        warnings.append("⚠️ Less than 2 CPU cores detected. Performance may be slow.")
    
    if requirements["available_memory_gb"] < 4:
        warnings.append("⚠️ Less than 4GB RAM available. Consider closing other applications.")
    
    if requirements["free_disk_gb"] < 5:
        warnings.append("⚠️ Less than 5GB disk space available. Models require significant storage.")
    
    requirements["warnings"] = warnings
    requirements["meets_requirements"] = len(warnings) == 0
    
    return requirements

def create_download_progress_bar():
    """Create a progress bar for downloads using Streamlit"""
    return st.progress(0)

def update_progress_bar(progress_bar, progress: float, status_text: str = ""):
    """Update progress bar with current progress"""
    progress_bar.progress(progress)
    if status_text:
        st.text(status_text)

def get_model_info(model_name: str) -> Dict[str, str]:
    """Get information about a model"""
    model_info = {
        "mistral-7b-instruct-v0.1.Q4_K_M.gguf": {
            "name": "Mistral 7B Instruct",
            "size": "~4.1 GB",
            "description": "Fast and efficient instruction-following model",
            "best_for": "General Q&A, instruction following",
            "memory_req": "6+ GB RAM"
        },
        "llama-2-7b-chat.Q4_K_M.gguf": {
            "name": "Llama 2 7B Chat", 
            "size": "~3.8 GB",
            "description": "Meta's conversational AI model",
            "best_for": "Chat, conversation, general tasks",
            "memory_req": "6+ GB RAM"
        },
        "codellama-7b-instruct.Q4_K_M.gguf": {
            "name": "Code Llama 7B Instruct",
            "size": "~3.8 GB", 
            "description": "Specialized for code understanding and generation",
            "best_for": "Technical documents, code-related questions",
            "memory_req": "6+ GB RAM"
        }
    }
    
    return model_info.get(model_name, {
        "name": "Unknown Model",
        "size": "Unknown",
        "description": "Model information not available",
        "best_for": "Unknown",
        "memory_req": "Unknown"
    })

def estimate_processing_time(num_pages: int, file_size_mb: float) -> str:
    """Estimate processing time based on document size"""
    # Rough estimates based on typical performance
    base_time_per_page = 0.5  # seconds per page
    size_factor = file_size_mb / 10  # additional time for larger files
    
    estimated_seconds = (num_pages * base_time_per_page) + size_factor
    
    if estimated_seconds < 60:
        return f"~{int(estimated_seconds)} seconds"
    elif estimated_seconds < 3600:
        minutes = int(estimated_seconds / 60)
        return f"~{minutes} minute{'s' if minutes > 1 else ''}"
    else:
        hours = int(estimated_seconds / 3600)
        return f"~{hours} hour{'s' if hours > 1 else ''}"

def get_pdf_metadata(file_path: str) -> Dict[str, Any]:
    """Extract metadata from PDF file using PyMuPDF"""
    try:
        doc = fitz.open(file_path)
        metadata = doc.metadata
        
        pdf_info = {
            'pages': len(doc),
            'file_size_mb': get_file_size_mb(file_path),
            'title': metadata.get('title', None),
            'author': metadata.get('author', None),
            'subject': metadata.get('subject', None),
            'creator': metadata.get('creator', None),
            'producer': metadata.get('producer', None),
            'creation_date': metadata.get('creationDate', None),
            'modification_date': metadata.get('modDate', None),
            'encrypted': doc.needs_pass,
            'text_extractable': True
        }
        
        # Test if text can be extracted from first page
        if len(doc) > 0:
            try:
                first_page = doc.load_page(0)
                test_text = first_page.get_text()
                pdf_info['text_extractable'] = bool(test_text.strip())
                
                # Estimate text density
                if test_text.strip():
                    char_count = len(test_text.replace(' ', '').replace('\n', ''))
                    pdf_info['text_density'] = char_count / len(doc)  # chars per page
                else:
                    pdf_info['text_density'] = 0
                    
            except Exception:
                pdf_info['text_extractable'] = False
                pdf_info['text_density'] = 0
        
        # Get estimated processing time
        pdf_info['estimated_processing_time'] = estimate_processing_time(
            pdf_info['pages'], 
            pdf_info['file_size_mb']
        )
        
        doc.close()
        return pdf_info
        
    except Exception as e:
        logging.getLogger(__name__).error(f"Failed to get metadata for {file_path}: {str(e)}")
        return {
            'pages': 0,
            'file_size_mb': get_file_size_mb(file_path),
            'error': str(e),
            'text_extractable': False
        }

def safe_filename(filename: str) -> str:
    """Create a safe filename by removing/replacing problematic characters"""
    import re
    # Remove problematic characters
    safe_name = re.sub(r'[<>:"/\\|?*]', '_', filename)
    # Limit length
    if len(safe_name) > 200:
        name, ext = os.path.splitext(safe_name)
        safe_name = name[:190] + ext
    return safe_name

def create_backup(file_path: str, backup_dir: str = "backups") -> Optional[str]:
    """Create a backup of a file"""
    try:
        backup_path = Path(backup_dir)
        backup_path.mkdir(exist_ok=True)
        
        filename = os.path.basename(file_path)
        timestamp = str(int(time.time()))
        backup_filename = f"{timestamp}_{filename}"
        backup_file_path = backup_path / backup_filename
        
        shutil.copy2(file_path, backup_file_path)
        return str(backup_file_path)
        
    except Exception as e:
        logging.getLogger(__name__).error(f"Failed to create backup: {str(e)}")
        return None

def display_system_info():
    """Display system information in Streamlit sidebar"""
    with st.sidebar:
        with st.expander("💻 System Info", expanded=False):
            sys_info = check_system_requirements()
            
            col1, col2 = st.columns(2)
            with col1:
                st.metric("CPU Cores", sys_info["cpu_cores"])
                st.metric("Memory (GB)", f"{sys_info['available_memory_gb']:.1f}/{sys_info['total_memory_gb']:.1f}")
            
            with col2:
                st.metric("Memory Usage", f"{sys_info['memory_percent']:.1f}%")
                st.metric("Disk Free (GB)", f"{sys_info['free_disk_gb']:.1f}")
            
            # Display warnings
            for warning in sys_info["warnings"]:
                st.warning(warning)
            
            if sys_info["meets_requirements"]:
                st.success("✅ System meets requirements")

def format_sources_for_display(sources: List[Dict[str, Any]], max_chars: int = 300) -> List[Dict[str, Any]]:
    """Format sources for better display in UI"""
    formatted_sources = []
    
    for i, source in enumerate(sources):
        # Truncate content if too long
        content = source.get('content', '')
        if len(content) > max_chars:
            content = content[:max_chars] + "..."
        
        # Format page info
        page_info = ""
        if source.get('page'):
            page_info = f"Page {source['page']}"
        
        # Format source name
        source_name = source.get('source', 'Unknown')
        if len(source_name) > 50:
            source_name = "..." + source_name[-47:]
        
        formatted_source = {
            'id': i + 1,
            'content': content,
            'page_info': page_info,
            'source_name': source_name,
            'score': source.get('score', 0.0),
            'original_content': source.get('content', '')
        }
        
        formatted_sources.append(formatted_source)
    
    return formatted_sources

# Error handling decorators
def handle_errors(func):
    """Decorator to handle common errors gracefully"""
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            logger = logging.getLogger(func.__module__)
            logger.error(f"Error in {func.__name__}: {str(e)}")
            st.error(f"An error occurred: {str(e)}")
            return None
    return wrapper

# Cache cleanup utility
def cleanup_cache():
    """Clean up various cache directories"""
    cache_dirs = [
        ".streamlit/cache",
        "__pycache__",
        ".pytest_cache",
        "logs",
    ]
    
    cleaned = []
    for cache_dir in cache_dirs:
        if os.path.exists(cache_dir):
            try:
                if os.path.isfile(cache_dir):
                    os.remove(cache_dir)
                else:
                    shutil.rmtree(cache_dir)
                cleaned.append(cache_dir)
            except Exception as e:
                logging.getLogger(__name__).warning(f"Could not clean {cache_dir}: {str(e)}")
    
    return cleaned