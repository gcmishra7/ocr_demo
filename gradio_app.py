"""
MLOps OCR Demo - Gradio Application

This is the main application that provides a user-friendly interface for
demonstrating the complete MLOps lifecycle for OCR applications.
"""

import os
import sys
import gradio as gr
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import json
import time
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / "src"))

from src.config.config import config
from src.utils.logger import setup_logger, get_logger
from src.data.data_processor import DataProcessor, DataVersioning
from src.models.custom_ocr_model import CustomOCRModel, ModelRegistry
from src.api.google_vision_api import GoogleVisionAPI, VisionAPIManager


# Set up logging
logger = setup_logger("gradio_app")
logger.info("Starting MLOps OCR Demo Application")


class MLOpsOCRDemo:
    """
    Main application class for the MLOps OCR Demo.
    
    Provides a comprehensive interface for demonstrating:
    - Data management and versioning
    - Custom model training and evaluation
    - Google Vision API integration
    - Model comparison and analysis
    - MLOps best practices
    """
    
    def __init__(self):
        """Initialize the MLOps OCR Demo application."""
        self.logger = get_logger("MLOpsOCRDemo")
        
        # Initialize components
        self.data_processor = DataProcessor()
        self.data_versioning = DataVersioning()
        self.custom_model = CustomOCRModel()
        self.model_registry = ModelRegistry()
        self.vision_api = GoogleVisionAPI()
        self.vision_manager = VisionAPIManager()
        
        # Initialize models
        self._initialize_models()
        
        self.logger.info("MLOps OCR Demo initialized successfully")
    
    def _initialize_models(self):
        """Initialize OCR models."""
        try:
            # Initialize custom model
            self.custom_model.initialize_model()
            self.logger.info("Custom OCR model initialized")
            
            # Initialize Vision API
            if self.vision_api.validate_api_access():
                self.logger.info("Google Vision API initialized")
            else:
                self.logger.warning("Google Vision API not available, using mock client")
                
        except Exception as e:
            self.logger.error("Failed to initialize models", error=str(e))
    
    def process_image_custom_model(self, image) -> Tuple[str, str, str]:
        """
        Process image using custom OCR model.
        
        Args:
            image: Input image (PIL Image or numpy array)
            
        Returns:
            Tuple of (extracted_text, confidence_info, processing_time)
        """
        try:
            start_time = time.time()
            
            # Convert image to numpy array if needed
            if isinstance(image, Image.Image):
                image_array = np.array(image)
            else:
                image_array = image
            
            # Perform OCR
            results = self.custom_model.predict(image_array)
            
            processing_time = time.time() - start_time
            
            # Format results
            if results:
                extracted_text = "\n".join([result['text'] for result in results])
                avg_confidence = np.mean([result['confidence'] for result in results])
                confidence_info = f"Average Confidence: {avg_confidence:.2%}\nDetected {len(results)} text blocks"
            else:
                extracted_text = "No text detected"
                confidence_info = "No text blocks found"
            
            time_info = f"Processing Time: {processing_time:.2f} seconds"
            
            self.logger.info("Custom model processing completed",
                           text_blocks=len(results),
                           processing_time=processing_time)
            
            return extracted_text, confidence_info, time_info
            
        except Exception as e:
            self.logger.error("Custom model processing failed", error=str(e))
            return f"Error: {str(e)}", "Processing failed", "N/A"
    
    def process_image_vision_api(self, image) -> Tuple[str, str, str]:
        """
        Process image using Google Vision API.
        
        Args:
            image: Input image (PIL Image or numpy array)
            
        Returns:
            Tuple of (extracted_text, confidence_info, processing_time)
        """
        try:
            start_time = time.time()
            
            # Convert image to bytes
            if isinstance(image, Image.Image):
                # Save to temporary file
                temp_path = Path("temp_image.jpg")
                image.save(temp_path)
            else:
                # Convert numpy array to PIL and save
                pil_image = Image.fromarray(image)
                temp_path = Path("temp_image.jpg")
                pil_image.save(temp_path)
            
            # Perform OCR using Vision API
            results = self.vision_api.detect_text(temp_path)
            
            processing_time = time.time() - start_time
            
            # Clean up temporary file
            if temp_path.exists():
                temp_path.unlink()
            
            # Format results
            if results:
                extracted_text = "\n".join([result.text for result in results])
                avg_confidence = np.mean([result.confidence for result in results])
                confidence_info = f"Average Confidence: {avg_confidence:.2%}\nDetected {len(results)} text blocks"
            else:
                extracted_text = "No text detected"
                confidence_info = "No text blocks found"
            
            time_info = f"Processing Time: {processing_time:.2f} seconds"
            
            self.logger.info("Vision API processing completed",
                           text_blocks=len(results),
                           processing_time=processing_time)
            
            return extracted_text, confidence_info, time_info
            
        except Exception as e:
            self.logger.error("Vision API processing failed", error=str(e))
            return f"Error: {str(e)}", "Processing failed", "N/A"
    
    def compare_models(self, image) -> Tuple[str, str, str]:
        """
        Compare results from both models.
        
        Args:
            image: Input image
            
        Returns:
            Tuple of (comparison_text, metrics_text, visualization_info)
        """
        try:
            # Process with both models
            custom_text, custom_conf, custom_time = self.process_image_custom_model(image)
            vision_text, vision_conf, vision_time = self.process_image_vision_api(image)
            
            # Create comparison
            comparison = f"""
## Model Comparison Results

### Custom OCR Model
**Text:** {custom_text}
**Confidence:** {custom_conf}
**Time:** {custom_time}

### Google Vision API
**Text:** {vision_text}
**Confidence:** {vision_conf}
**Time:** {vision_time}
"""
            
            # Calculate metrics
            custom_blocks = len(custom_text.split('\n')) if custom_text != "No text detected" else 0
            vision_blocks = len(vision_text.split('\n')) if vision_text != "No text detected" else 0
            
            metrics = f"""
## Performance Metrics

- **Custom Model Text Blocks:** {custom_blocks}
- **Vision API Text Blocks:** {vision_blocks}
- **Processing Speed:** {'Custom Model' if '0.2' in custom_time else 'Vision API'} is faster
- **Text Detection:** {'Custom Model' if custom_blocks > vision_blocks else 'Vision API'} detected more text
"""
            
            # Create visualization info
            viz_info = f"""
## Visualization

The comparison shows the differences between:
1. **Custom Model:** Local processing, customizable, no API costs
2. **Vision API:** Cloud-based, high accuracy, pay-per-use

Choose based on your requirements for accuracy, speed, and cost.
"""
            
            return comparison, metrics, viz_info
            
        except Exception as e:
            self.logger.error("Model comparison failed", error=str(e))
            return f"Error: {str(e)}", "Comparison failed", "N/A"
    
    def demonstrate_mlops_pipeline(self) -> str:
        """
        Demonstrate the complete MLOps pipeline.
        
        Returns:
            Pipeline demonstration text
        """
        try:
            pipeline_demo = """
# 🚀 MLOps Pipeline Demonstration

## 1. **Data Management & Versioning** 📊
- ✅ Data validation and quality checks
- ✅ Data versioning with DVC
- ✅ Reproducible data pipelines

## 2. **Model Development** 🧠
- ✅ Custom OCR model training
- ✅ Model evaluation and validation
- ✅ Hyperparameter optimization

## 3. **Model Registry** 📝
- ✅ Model versioning with MLflow
- ✅ Model metadata tracking
- ✅ Model deployment management

## 4. **API Integration** 🌐
- ✅ Google Vision API integration
- ✅ Custom model API development
- ✅ Performance monitoring

## 5. **Testing & Validation** 🧪
- ✅ Unit tests for components
- ✅ Integration tests for pipelines
- ✅ Model performance validation

## 6. **CI/CD Pipeline** ⚙️
- ✅ Automated testing
- ✅ Automated deployment
- ✅ Continuous monitoring

## 7. **Monitoring & Logging** 📈
- ✅ Structured logging
- ✅ Performance metrics
- ✅ Error tracking

---

**This demo showcases industry-standard MLOps practices!**
"""
            
            self.logger.info("MLOps pipeline demonstration completed")
            return pipeline_demo
            
        except Exception as e:
            self.logger.error("Pipeline demonstration failed", error=str(e))
            return f"Error: {str(e)}"
    
    def show_model_registry(self) -> str:
        """
        Show model registry information.
        
        Returns:
            Model registry information
        """
        try:
            models = self.model_registry.list_models()
            
            if models:
                registry_info = "## 📋 Model Registry\n\n"
                for model in models:
                    registry_info += f"""
### Model: {model['model_name']} v{model['version']}
- **Path:** {model['model_path']}
- **Accuracy:** {model['metrics']['accuracy']:.2%}
- **F1 Score:** {model['metrics']['f1_score']:.2%}
- **Inference Time:** {model['metrics']['inference_time']:.2f}s
- **Registered:** {model['registered_at']}
- **Tags:** {', '.join(model['tags'].keys()) if model['tags'] else 'None'}

---
"""
            else:
                registry_info = """
## 📋 Model Registry

No models currently registered.

To register a model:
1. Train a custom OCR model
2. Evaluate its performance
3. Register it in the model registry
4. Deploy to production
"""
            
            self.logger.info("Model registry information displayed")
            return registry_info
            
        except Exception as e:
            self.logger.error("Failed to show model registry", error=str(e))
            return f"Error: {str(e)}"
    
    def show_data_versioning(self) -> str:
        """
        Show data versioning information.
        
        Returns:
            Data versioning information
        """
        try:
            versions = self.data_versioning.get_data_versions()
            
            if versions:
                versioning_info = "## 📁 Data Versioning (DVC)\n\n"
                for version in versions:
                    versioning_info += f"""
### Version: {version['version']}
- **Commit Hash:** {version['commit_hash']}
- **Created:** {version['created_at']}
- **File Count:** {version['file_count']}

---
"""
            else:
                versioning_info = """
## 📁 Data Versioning (DVC)

No data versions currently tracked.

To set up data versioning:
1. Initialize DVC in your project
2. Add data files to DVC
3. Commit data versions
4. Track data lineage
"""
            
            self.logger.info("Data versioning information displayed")
            return versioning_info
            
        except Exception as e:
            self.logger.error("Failed to show data versioning", error=str(e))
            return f"Error: {str(e)}"


def create_gradio_interface():
    """Create the Gradio interface for the MLOps OCR Demo."""
    
    # Initialize the demo application
    demo_app = MLOpsOCRDemo()
    
    # Create the Gradio interface
    with gr.Blocks(
        title="MLOps OCR Demo",
        theme=gr.themes.Soft(),
        css="""
        .gradio-container {
            max-width: 1200px !important;
        }
        .main-header {
            text-align: center;
            padding: 20px;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        """
    ) as interface:
        
        # Header
        gr.HTML("""
        <div class="main-header">
            <h1>🔍 MLOps OCR Demo</h1>
            <p>Complete MLOps Lifecycle for Optical Character Recognition</p>
            <p>Demonstrating Reproducibility, Versioning, Automation, Testing, and CI/CD</p>
        </div>
        """)
        
        # Main tabs
        with gr.Tabs():
            
            # Tab 1: OCR Processing
            with gr.TabItem("🔍 OCR Processing"):
                gr.Markdown("""
                ## OCR Processing Demo
                
                Upload an image to see how different OCR models perform:
                - **Custom Model:** Local EasyOCR-based model
                - **Google Vision API:** Cloud-based OCR service
                """)
                
                with gr.Row():
                    with gr.Column():
                        input_image = gr.Image(
                            label="Upload Image",
                            type="pil",
                            height=400
                        )
                        
                        with gr.Row():
                            custom_btn = gr.Button("Process with Custom Model", variant="primary")
                            vision_btn = gr.Button("Process with Vision API", variant="secondary")
                            compare_btn = gr.Button("Compare Both Models", variant="primary")
                    
                    with gr.Column():
                        output_text = gr.Textbox(
                            label="Extracted Text",
                            lines=10,
                            max_lines=15
                        )
                        
                        confidence_info = gr.Textbox(
                            label="Confidence & Metrics",
                            lines=3
                        )
                        
                        processing_time = gr.Textbox(
                            label="Processing Information",
                            lines=2
                        )
                
                # Event handlers
                custom_btn.click(
                    demo_app.process_image_custom_model,
                    inputs=[input_image],
                    outputs=[output_text, confidence_info, processing_time]
                )
                
                vision_btn.click(
                    demo_app.process_image_vision_api,
                    inputs=[input_image],
                    outputs=[output_text, confidence_info, processing_time]
                )
                
                compare_btn.click(
                    demo_app.compare_models,
                    inputs=[input_image],
                    outputs=[output_text, confidence_info, processing_time]
                )
            
            # Tab 2: MLOps Pipeline
            with gr.TabItem("⚙️ MLOps Pipeline"):
                gr.Markdown("""
                ## MLOps Pipeline Demonstration
                
                This section shows the complete MLOps lifecycle for OCR applications.
                """)
                
                pipeline_btn = gr.Button("Show MLOps Pipeline", variant="primary")
                pipeline_output = gr.Markdown(label="Pipeline Information")
                
                pipeline_btn.click(
                    demo_app.demonstrate_mlops_pipeline,
                    outputs=[pipeline_output]
                )
            
            # Tab 3: Model Registry
            with gr.TabItem("📋 Model Registry"):
                gr.Markdown("""
                ## Model Registry
                
                Track and manage different versions of your OCR models.
                """)
                
                registry_btn = gr.Button("Show Model Registry", variant="primary")
                registry_output = gr.Markdown(label="Registry Information")
                
                registry_btn.click(
                    demo_app.show_model_registry,
                    outputs=[registry_output]
                )
            
            # Tab 4: Data Versioning
            with gr.TabItem("📁 Data Versioning"):
                gr.Markdown("""
                ## Data Versioning (DVC)
                
                Track different versions of your training data for reproducibility.
                """)
                
                versioning_btn = gr.Button("Show Data Versions", variant="primary")
                versioning_output = gr.Markdown(label="Versioning Information")
                
                versioning_btn.click(
                    demo_app.show_data_versioning,
                    outputs=[versioning_output]
                )
            
            # Tab 5: About
            with gr.TabItem("ℹ️ About"):
                gr.Markdown("""
                ## About This Demo
                
                This application demonstrates a complete MLOps lifecycle for building OCR applications.
                
                ### 🎯 Learning Objectives
                
                **1. Reproducibility**
                - Version-controlled data, models, and configurations
                - Docker containers for consistent environments
                - Requirements.txt and environment management
                
                **2. Versioning**
                - Git for code versioning
                - DVC for data and model versioning
                - Model registry with MLflow
                - Configuration management with YAML
                
                **3. Automation**
                - Automated training pipelines
                - Automated evaluation and testing
                - Automated deployment workflows
                
                **4. Testing**
                - Unit tests for components
                - Integration tests for pipelines
                - Model performance validation
                - API endpoint testing
                
                **5. CI/CD**
                - GitHub Actions for continuous integration
                - Automated testing and deployment
                - Model deployment automation
                
                ### 🔧 Technologies Used
                
                - **Python:** Core programming language
                - **Docker:** Containerization
                - **GitHub Actions:** CI/CD
                - **MLflow:** Model registry and tracking
                - **DVC:** Data versioning
                - **Gradio:** Web interface
                - **Google Vision API:** Cloud OCR service
                - **EasyOCR:** Custom OCR model
                - **Pytest:** Testing framework
                - **YAML:** Configuration management
                
                ### 📚 Educational Value
                
                This demo provides hands-on experience with:
                - **Real-world MLOps practices**
                - **Industry-standard tools**
                - **Best practices** for model development and deployment
                - **Practical implementation** of theoretical concepts
                
                ### 🚀 Getting Started
                
                1. **Explore the code structure**
                2. **Run the demo application**
                3. **Modify and experiment with different components**
                4. **Add your own custom models**
                5. **Extend the MLOps pipeline**
                
                ---
                
                **Happy Learning! 🎉**
                """)
    
    return interface


if __name__ == "__main__":
    # Create and launch the interface
    interface = create_gradio_interface()
    
    # Launch the application
    interface.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,  # Create public link
        debug=True,
        show_error=True
    ) 