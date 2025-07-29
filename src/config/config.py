"""
Configuration Management Module

This module handles all configuration settings for the MLOps OCR demo.
Uses Hydra for configuration management and environment variable handling.
"""

import os
from dataclasses import dataclass
from typing import Optional
from pathlib import Path
import yaml
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


@dataclass
class GoogleCloudConfig:
    """Google Cloud configuration settings."""
    credentials_path: str = os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "")
    project_id: str = os.getenv("GOOGLE_CLOUD_PROJECT", "")
    vision_api_enabled: bool = True


@dataclass
class MLflowConfig:
    """MLflow configuration settings."""
    tracking_uri: str = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    experiment_name: str = os.getenv("MLFLOW_EXPERIMENT_NAME", "ocr_demo")
    registry_path: str = os.getenv("MODEL_REGISTRY_PATH", "./models/registry")


@dataclass
class DataConfig:
    """Data configuration settings."""
    data_path: Path = Path(os.getenv("DATA_PATH", "./data"))
    raw_data_path: Path = Path(os.getenv("RAW_DATA_PATH", "./data/raw"))
    processed_data_path: Path = Path(os.getenv("PROCESSED_DATA_PATH", "./data/processed"))
    test_data_path: Path = Path(os.getenv("TEST_DATA_PATH", "./tests/data"))


@dataclass
class ModelConfig:
    """Model configuration settings."""
    custom_model_path: Path = Path(os.getenv("CUSTOM_MODEL_PATH", "./models/custom"))
    model_registry_path: Path = Path(os.getenv("MODEL_REGISTRY_PATH", "./models/registry"))
    supported_languages: list = None
    
    def __post_init__(self):
        if self.supported_languages is None:
            self.supported_languages = ["en", "es", "fr", "de"]


@dataclass
class APIConfig:
    """API configuration settings."""
    host: str = os.getenv("API_HOST", "0.0.0.0")
    port: int = int(os.getenv("API_PORT", "8000"))
    workers: int = int(os.getenv("API_WORKERS", "4"))
    debug: bool = False


@dataclass
class LoggingConfig:
    """Logging configuration settings."""
    level: str = os.getenv("LOG_LEVEL", "INFO")
    log_file: Path = Path(os.getenv("LOG_FILE", "./logs/ocr_demo.log"))
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


@dataclass
class DockerConfig:
    """Docker configuration settings."""
    image_name: str = os.getenv("DOCKER_IMAGE_NAME", "ocr-demo")
    tag: str = os.getenv("DOCKER_TAG", "latest")
    dockerfile_path: Path = Path("./docker/Dockerfile")


class Config:
    """Main configuration class that combines all configuration sections."""
    
    def __init__(self):
        self.google_cloud = GoogleCloudConfig()
        self.mlflow = MLflowConfig()
        self.data = DataConfig()
        self.model = ModelConfig()
        self.api = APIConfig()
        self.logging = LoggingConfig()
        self.docker = DockerConfig()
        
        # Create necessary directories
        self._create_directories()
    
    def _create_directories(self):
        """Create necessary directories if they don't exist."""
        directories = [
            self.data.data_path,
            self.data.raw_data_path,
            self.data.processed_data_path,
            self.data.test_data_path,
            self.model.custom_model_path,
            self.model.model_registry_path,
            self.logging.log_file.parent
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def save_config(self, file_path: Path):
        """Save configuration to YAML file."""
        config_dict = {
            "google_cloud": self.google_cloud.__dict__,
            "mlflow": self.mlflow.__dict__,
            "data": {k: str(v) for k, v in self.data.__dict__.items()},
            "model": {k: str(v) if isinstance(v, Path) else v for k, v in self.model.__dict__.items()},
            "api": self.api.__dict__,
            "logging": {k: str(v) if isinstance(v, Path) else v for k, v in self.logging.__dict__.items()},
            "docker": {k: str(v) if isinstance(v, Path) else v for k, v in self.docker.__dict__.items()}
        }
        
        with open(file_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)
    
    @classmethod
    def load_config(cls, file_path: Path):
        """Load configuration from YAML file."""
        with open(file_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        config = cls()
        # Update configuration with loaded values
        # This is a simplified version - in practice, you'd want more robust loading
        return config


# Global configuration instance
config = Config() 