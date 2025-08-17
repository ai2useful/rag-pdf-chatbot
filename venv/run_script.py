#!/usr/bin/env python3
"""
Startup script for RAG PDF Chatbot
Handles environment setup and application launch
"""

import os
import sys
import subprocess
import argparse
import json
import logging
from pathlib import Path

def load_config():
    """Load configuration from config.json"""
    try:
        with open("config.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        print("⚠️ config.json not found. Using default settings.")
        return {}
    except json.JSONDecodeError as e:
        print(f"⚠️ Error reading config.json: {e}")
        return {}

def setup_environment(config):
    """Setup environment variables"""
    # Streamlit configuration
    os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
    os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"
    
    # Set theme if specified
    ui_config = config.get("ui_config", {})
    if ui_config.get("theme"):
        os.environ["STREAMLIT_THEME_BASE"] = ui_config["theme"]

def create_directories(config):
    """Create necessary directories"""
    paths = config.get("paths", {
        "upload_dir": "uploaded_pdfs",
        "vector_store_dir": "vector_store", 
        "models_dir": "models",
        "logs_dir": "logs",
        "backups_dir": "backups"
    })
    
    for path_name, path_value in paths.items():
        Path(path_value).mkdir(exist_ok=True)
        print(f"✓ Directory ready: {path_value}")

def check_dependencies():
    """Check if required dependencies are installed"""
    required_packages = [
        "streamlit",
        "PyPDF2", 
        "sentence_transformers",
        "faiss-cpu",
        "llama_cpp"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing packages: {', '.join(missing_packages)}")
        print("Run: pip install -r requirements.txt")
        return False
    
    print("✓ All dependencies installed")
    return True

def run_app(host="localhost", port=8501, dev_mode=False):
    """Run the Streamlit application"""
    
    cmd = [
        sys.executable, "-m", "streamlit", "run", "app.py",
        "--server.address", host,
        "--server.port", str(port)
    ]
    
    if not dev_mode:
        cmd.extend([
            "--server.headless", "true",
            "--browser.gatherUsageStats", "false"
        ])
    
    print(f"🚀 Starting RAG PDF Chatbot on http://{host}:{port}")
    print("Press Ctrl+C to stop the server")
    
    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\n👋 Shutting down gracefully...")
    except Exception as e:
        print(f"❌ Error running application: {e}")

def main():
    parser = argparse.ArgumentParser(description="RAG PDF Chatbot Launcher")
    
    parser.add_argument(
        "--host", 
        default="localhost", 
        help="Host address (default: localhost)"
    )
    
    parser.add_argument(
        "--port", 
        type=int, 
        default=8501, 
        help="Port number (default: 8501)"
    )
    
    parser.add_argument(
        "--dev", 
        action="store_true", 
        help="Run in development mode (opens browser)"
    )
    
    parser.add_argument(
        "--setup", 
        action="store_true", 
        help="Run setup only (don't start the app)"
    )
    
    parser.add_argument(
        "--check", 
        action="store_true", 
        help="Check dependencies and exit"
    )
    
    parser.add_argument(
        "--config", 
        default="config.json", 
        help="Configuration file path"
    )
    
    args = parser.parse_args()
    
    print("📚 RAG PDF Chatbot Launcher")
    print("=" * 40)
    
    # Load configuration
    config = load_config()
    
    # Setup environment
    setup_environment(config)
    
    # Create directories
    create_directories(config)
    
    # Check dependencies
    if not check_dependencies():
        if args.check:
            sys.exit(1)
        
        print("\n💡 Tip: Run setup.py first to install dependencies")
        response = input("Do you want to install dependencies now? (y/n): ")
        if response.lower().startswith('y'):
            try:
                subprocess.check_call([sys.executable, "setup.py"])
                print("✓ Setup completed")
            except subprocess.CalledProcessError:
                print("❌ Setup failed")
                sys.exit(1)
        else:
            sys.exit(1)
    
    if args.check:
        print("✓ All checks passed")
        sys.exit(0)
    
    if args.setup:
        print("✓ Setup completed")
        sys.exit(0)
    
    # Run the application
    run_app(host=args.host, port=args.port, dev_mode=args.dev)

if __name__ == "__main__":
    main()