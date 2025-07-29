#!/usr/bin/env python3
"""
MLOps OCR Demo - Run Script

This script provides an easy way to run the MLOps OCR demonstration.
It handles setup, validation, and launching the application.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def check_dependencies():
    """Check if required dependencies are installed."""
    required_packages = [
        'gradio',
        'opencv-python',
        'numpy',
        'pandas',
        'easyocr',
        'mlflow'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing required packages: {', '.join(missing_packages)}")
        print("Please install them using: pip install -r requirements.txt")
        return False
    
    print("✅ All required dependencies are installed")
    return True

def setup_environment():
    """Set up the environment for the demo."""
    print("🔧 Setting up environment...")
    
    # Create necessary directories
    directories = [
        'data/raw',
        'data/processed',
        'models/custom',
        'models/registry',
        'logs'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
    
    print("✅ Environment setup completed")

def validate_configuration():
    """Validate the configuration."""
    print("🔍 Validating configuration...")
    
    # Check if environment file exists
    env_file = Path('.env')
    if not env_file.exists():
        print("⚠️  No .env file found. Using default configuration.")
        print("   Copy env.example to .env and configure as needed.")
    else:
        print("✅ Environment configuration found")
    
    # Check MLflow configuration
    mlflow_uri = os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000')
    print(f"📊 MLflow tracking URI: {mlflow_uri}")
    
    # Check Google Cloud configuration
    google_creds = os.getenv('GOOGLE_APPLICATION_CREDENTIALS', '')
    if google_creds:
        print("✅ Google Cloud credentials configured")
    else:
        print("⚠️  Google Cloud credentials not configured. Vision API will use mock client.")

def start_mlflow_server():
    """Start MLflow server if not already running."""
    try:
        import mlflow
        mlflow.set_tracking_uri("http://localhost:5000")
        mlflow.get_experiment_by_name("ocr_demo")
        print("✅ MLflow server is running")
        return True
    except:
        print("🚀 Starting MLflow server...")
        try:
            subprocess.Popen([
                sys.executable, "-m", "mlflow", "server",
                "--host", "0.0.0.0",
                "--port", "5000"
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print("✅ MLflow server started")
            return True
        except Exception as e:
            print(f"⚠️  Could not start MLflow server: {e}")
            print("   The demo will work without MLflow tracking")
            return False

def run_demo(mode='local', port=7860, share=False):
    """Run the MLOps OCR demo."""
    print(f"🎯 Starting MLOps OCR Demo in {mode} mode...")
    
    # Import and run the demo
    try:
        from gradio_app import create_gradio_interface
        
        interface = create_gradio_interface()
        
        # Launch the interface
        interface.launch(
            server_name="0.0.0.0",
            server_port=port,
            share=share,
            debug=(mode == 'development'),
            show_error=True
        )
        
    except Exception as e:
        print(f"❌ Failed to start demo: {e}")
        print("Please check the error message above and try again.")
        return False
    
    return True

def run_tests():
    """Run the test suite."""
    print("🧪 Running tests...")
    
    try:
        result = subprocess.run([
            sys.executable, "-m", "pytest", "tests/", "-v"
        ], capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ All tests passed")
            return True
        else:
            print("❌ Some tests failed")
            print(result.stdout)
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ Failed to run tests: {e}")
        return False

def main():
    """Main function."""
    parser = argparse.ArgumentParser(description="MLOps OCR Demo")
    parser.add_argument(
        '--mode',
        choices=['local', 'development', 'production'],
        default='local',
        help='Run mode (default: local)'
    )
    parser.add_argument(
        '--port',
        type=int,
        default=7860,
        help='Port to run the demo on (default: 7860)'
    )
    parser.add_argument(
        '--share',
        action='store_true',
        help='Create a public link for the demo'
    )
    parser.add_argument(
        '--test',
        action='store_true',
        help='Run tests before starting the demo'
    )
    parser.add_argument(
        '--setup-only',
        action='store_true',
        help='Only run setup, don\'t start the demo'
    )
    
    args = parser.parse_args()
    
    print("🚀 MLOps OCR Demo - Setup and Launch")
    print("=" * 50)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Setup environment
    setup_environment()
    
    # Validate configuration
    validate_configuration()
    
    # Start MLflow server
    start_mlflow_server()
    
    # Run tests if requested
    if args.test:
        if not run_tests():
            print("❌ Tests failed. Exiting.")
            sys.exit(1)
    
    # Setup only mode
    if args.setup_only:
        print("✅ Setup completed successfully!")
        return
    
    # Run the demo
    print("\n" + "=" * 50)
    print("🎉 Starting MLOps OCR Demo...")
    print("=" * 50)
    
    success = run_demo(
        mode=args.mode,
        port=args.port,
        share=args.share
    )
    
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main() 