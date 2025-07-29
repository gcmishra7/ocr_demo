# MLOps OCR Demo - Quick Start Guide

## 🚀 Get Started in 5 Minutes

This guide helps you quickly set up and run the MLOps OCR demo to understand MLOps concepts in practice.

## 📋 Prerequisites

- Python 3.8+
- Docker (optional)
- Git

## ⚡ Quick Setup

### 1. Clone and Install
```bash
git clone <your-repo-url>
cd ocr_demo
pip install -r requirements.txt
```

### 2. Configure Environment
```bash
cp env.example .env
# Edit .env with your settings (optional for basic demo)
```

### 3. Run the Demo
```bash
python run_demo.py
```

### 4. Access the Application
- **Gradio UI**: http://localhost:7860
- **MLflow UI**: http://localhost:5000

## 🎯 What You'll Learn

### MLOps Lifecycle Components

1. **Reproducibility** 🔄
   - Fixed random seeds
   - Version-controlled dependencies
   - Consistent environments

2. **Versioning** 📝
   - Git for code
   - DVC for data
   - MLflow for models

3. **Automation** ⚙️
   - Automated data processing
   - Automated model training
   - Automated testing

4. **Testing** 🧪
   - Unit tests
   - Integration tests
   - Performance tests

5. **CI/CD** 🚀
   - GitHub Actions pipeline
   - Automated deployment
   - Quality gates

## 🔧 Key Features

### Two OCR Approaches
- **Custom Model**: EasyOCR-based implementation
- **Google Vision API**: Cloud-based OCR service

### MLOps Tools Integration
- **MLflow**: Experiment tracking and model registry
- **DVC**: Data versioning
- **Docker**: Containerization
- **GitHub Actions**: CI/CD pipeline

### Monitoring & Observability
- **Prometheus**: Metrics collection
- **Grafana**: Visualization
- **Structured Logging**: Comprehensive logging

## 📊 Project Structure

```
ocr_demo/
├── src/                    # Source code
│   ├── data/              # Data processing
│   ├── models/            # Model implementations
│   ├── api/               # API integrations
│   ├── utils/             # Utilities
│   └── config/            # Configuration
├── tests/                 # Test suite
├── docker/                # Docker configurations
├── .github/               # CI/CD workflows
├── docs/                  # Documentation
├── gradio_app.py          # Main application
└── run_demo.py           # Setup script
```

## 🎮 Interactive Demo

### Upload and Process Images
1. Open http://localhost:7860
2. Upload an image with text
3. Click "Custom Model" or "Google Vision API"
4. View extracted text and confidence scores

### Explore MLOps Components
- **Data Processing**: See how images are preprocessed
- **Model Comparison**: Compare different OCR approaches
- **Experiment Tracking**: View MLflow experiments
- **Monitoring**: Check metrics and logs

## 🔍 Understanding the Code

### Data Processing Pipeline
```python
# src/data/data_processor.py
class DataProcessor:
    def preprocess_image(self, image_path):
        # Load, resize, normalize image
        # Validate data quality
        # Return processed image
```

### Model Training with MLflow
```python
# src/models/custom_ocr_model.py
with mlflow.start_run():
    mlflow.log_params(params)
    model = train_model(data, params)
    mlflow.log_metrics(metrics)
    mlflow.log_model(model, "ocr_model")
```

### API Integration
```python
# src/api/google_vision_api.py
class GoogleVisionAPI:
    def detect_text(self, image_path):
        # Call Google Vision API
        # Handle responses and errors
        # Return structured results
```

## 🧪 Testing the System

### Run Tests
```bash
# Unit tests
pytest tests/test_data_processor.py -v

# Integration tests
pytest tests/test_integration.py -v

# All tests with coverage
pytest tests/ -v --cov=src --cov-report=html
```

### Test Different Scenarios
- **Valid Images**: Test with clear text images
- **Invalid Images**: Test error handling
- **Performance**: Test with large images
- **API Limits**: Test rate limiting

## 🐳 Docker Deployment

### Build and Run
```bash
# Build image
docker build -f docker/Dockerfile -t ocr-demo .

# Run with Docker Compose
docker-compose up -d

# Check services
docker-compose ps
```

### Access Services
- **Application**: http://localhost:7860
- **MLflow**: http://localhost:5000
- **Grafana**: http://localhost:3000

## 📈 Monitoring Dashboard

### Key Metrics
- **Prediction Count**: Total OCR predictions
- **Response Time**: Model inference latency
- **Accuracy**: Model performance metrics
- **Error Rate**: Failed predictions

### View Metrics
1. Open Grafana: http://localhost:3000
2. Login: admin/admin
3. Navigate to OCR Demo dashboard
4. View real-time metrics

## 🔄 CI/CD Pipeline

### Automated Workflow
1. **Code Quality**: Linting, type checking
2. **Testing**: Unit and integration tests
3. **Security**: Vulnerability scanning
4. **Building**: Docker image creation
5. **Deployment**: Automated deployment

### Trigger Pipeline
```bash
git add .
git commit -m "Update model parameters"
git push origin main
# Check GitHub Actions for pipeline status
```

## 🎓 Learning Path

### Beginner Level
1. Run the demo and explore the UI
2. Understand the project structure
3. Read the documentation
4. Try different image types

### Intermediate Level
1. Modify model parameters
2. Add new data processing steps
3. Implement custom tests
4. Extend the API functionality

### Advanced Level
1. Add new ML models
2. Implement custom CI/CD steps
3. Add monitoring alerts
4. Scale with Kubernetes

## 🚨 Troubleshooting

### Common Issues

**Port Already in Use**
```bash
# Find and kill process
lsof -ti:7860 | xargs kill -9
```

**MLflow Connection Error**
```bash
# Start MLflow server
mlflow server --host 0.0.0.0 --port 5000 &
```

**Docker Build Fails**
```bash
# Clean and rebuild
docker system prune -a
docker build --no-cache -f docker/Dockerfile -t ocr-demo .
```

**Import Errors**
```bash
# Install missing dependencies
pip install -r requirements.txt
export PYTHONPATH=$PYTHONPATH:$(pwd)
```

## 📚 Next Steps

### Explore Further
1. **Read the main documentation**: `docs/README.md`
2. **Study the lifecycle guide**: `docs/mlops-lifecycle.md`
3. **Follow the practical guide**: `docs/practical-implementation.md`

### Extend the Demo
1. Add new OCR models
2. Implement data augmentation
3. Add model explainability
4. Create custom dashboards

### Join the Community
- GitHub Issues: Report bugs and request features
- Discussions: Share ideas and solutions
- Contributing: Submit pull requests

## 🎉 Success!

You've successfully set up and run the MLOps OCR demo! This provides a practical foundation for understanding MLOps concepts and implementing them in your own projects.

**Remember**: MLOps is about creating reliable, scalable, and maintainable machine learning systems. Start small, iterate often, and always monitor your systems.

---

**Happy MLOps Learning! 🚀** 