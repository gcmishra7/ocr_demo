"""
Unit tests for the data processor module.

This module demonstrates testing best practices for MLOps components.
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import shutil
from unittest.mock import Mock, patch

from src.data.data_processor import DataProcessor, DataValidationResult, DataVersioning


class TestDataProcessor:
    """Test cases for DataProcessor class."""
    
    @pytest.fixture
    def data_processor(self):
        """Create a DataProcessor instance for testing."""
        return DataProcessor()
    
    @pytest.fixture
    def sample_image(self):
        """Create a sample image for testing."""
        # Create a simple test image
        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        return image
    
    @pytest.fixture
    def temp_image_file(self, sample_image):
        """Create a temporary image file for testing."""
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as f:
            # In a real test, you'd save the image properly
            # For demo purposes, we'll just create a file
            f.write(b'fake image data')
            temp_path = f.name
        
        yield Path(temp_path)
        
        # Cleanup
        Path(temp_path).unlink(missing_ok=True)
    
    def test_data_processor_initialization(self, data_processor):
        """Test DataProcessor initialization."""
        assert data_processor is not None
        assert data_processor.supported_formats == ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        assert data_processor.min_image_size == (100, 100)
        assert data_processor.max_image_size == (2048, 2048)
    
    def test_load_image_valid_path(self, data_processor, temp_image_file):
        """Test loading a valid image."""
        # Mock cv2.imread to return a valid image
        with patch('cv2.imread') as mock_imread:
            mock_image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
            mock_imread.return_value = mock_image
            
            with patch('cv2.cvtColor') as mock_cvtcolor:
                mock_cvtcolor.return_value = mock_image
                
                result = data_processor.load_image(temp_image_file)
                
                assert result is not None
                assert result.shape == (100, 100, 3)
                mock_imread.assert_called_once_with(str(temp_image_file))
    
    def test_load_image_invalid_path(self, data_processor):
        """Test loading an image with invalid path."""
        invalid_path = Path("/nonexistent/image.jpg")
        
        with pytest.raises(FileNotFoundError):
            data_processor.load_image(invalid_path)
    
    def test_load_image_unsupported_format(self, data_processor):
        """Test loading an image with unsupported format."""
        with tempfile.NamedTemporaryFile(suffix='.gif', delete=False) as f:
            temp_path = Path(f.name)
        
        try:
            with pytest.raises(ValueError, match="Unsupported image format"):
                data_processor.load_image(temp_path)
        finally:
            temp_path.unlink(missing_ok=True)
    
    def test_preprocess_image(self, data_processor, sample_image):
        """Test image preprocessing."""
        target_size = (256, 256)
        
        with patch('cv2.resize') as mock_resize:
            mock_resize.return_value = np.random.random((256, 256, 3))
            
            with patch('cv2.cvtColor') as mock_cvtcolor:
                mock_cvtcolor.return_value = np.random.random((256, 256))
                
                with patch('cv2.equalizeHist') as mock_equalize:
                    mock_equalize.return_value = np.random.randint(0, 255, (256, 256))
                    
                    result = data_processor.preprocess_image(sample_image, target_size)
                    
                    assert result is not None
                    assert result.shape == (256, 256)
                    assert result.dtype == np.float32
                    assert np.all(result >= 0) and np.all(result <= 1)
    
    def test_validate_data_valid_directory(self, data_processor):
        """Test data validation with valid directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create some test image files
            for i in range(3):
                test_file = temp_path / f"test_image_{i}.jpg"
                test_file.write_bytes(b'fake image data')
            
            # Mock the load_image method to avoid actual image processing
            with patch.object(data_processor, 'load_image') as mock_load:
                mock_load.return_value = np.random.randint(0, 255, (200, 200, 3))
                
                result = data_processor.validate_data(temp_path)
                
                assert isinstance(result, DataValidationResult)
                assert result.is_valid is True
                assert result.metadata['total_files'] == 3
                assert result.metadata['valid_files'] == 3
                assert result.metadata['invalid_files'] == 0
                assert len(result.errors) == 0
    
    def test_validate_data_invalid_directory(self, data_processor):
        """Test data validation with invalid directory."""
        invalid_path = Path("/nonexistent/directory")
        
        result = data_processor.validate_data(invalid_path)
        
        assert isinstance(result, DataValidationResult)
        assert result.is_valid is False
        assert len(result.errors) > 0
        assert "does not exist" in result.errors[0]
    
    def test_validate_data_with_warnings(self, data_processor):
        """Test data validation with warnings (small images)."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create a test image file
            test_file = temp_path / "small_image.jpg"
            test_file.write_bytes(b'fake image data')
            
            # Mock the load_image method to return a small image
            with patch.object(data_processor, 'load_image') as mock_load:
                mock_load.return_value = np.random.randint(0, 255, (50, 50, 3))
                
                result = data_processor.validate_data(temp_path)
                
                assert isinstance(result, DataValidationResult)
                assert result.is_valid is True
                assert len(result.warnings) > 0
                assert "too small" in result.warnings[0]
    
    def test_create_augmentation_pipeline(self, data_processor):
        """Test augmentation pipeline creation."""
        pipeline = data_processor.create_augmentation_pipeline()
        
        assert pipeline is not None
        # Check that it's an Albumentations Compose object
        assert hasattr(pipeline, 'transforms')
    
    def test_save_metadata(self, data_processor):
        """Test metadata saving."""
        metadata = {
            'test_key': 'test_value',
            'number': 42,
            'list': [1, 2, 3]
        }
        
        with tempfile.TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "metadata.json"
            
            data_processor.save_metadata(metadata, output_path)
            
            assert output_path.exists()
            
            # Verify the saved content
            import json
            with open(output_path, 'r') as f:
                saved_metadata = json.load(f)
            
            assert saved_metadata['test_key'] == 'test_value'
            assert saved_metadata['number'] == 42
            assert saved_metadata['list'] == [1, 2, 3]
    
    def test_create_data_version(self, data_processor):
        """Test data version creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            source_path = Path(temp_dir) / "source"
            source_path.mkdir()
            
            # Create some test files
            for i in range(2):
                test_file = source_path / f"test_file_{i}.txt"
                test_file.write_text(f"test content {i}")
            
            # Mock the processed data path
            with patch.object(data_processor, 'config') as mock_config:
                mock_config.data.processed_data_path = Path(temp_dir) / "processed"
                
                versioned_path = data_processor.create_data_version(source_path, "v1.0")
                
                assert versioned_path.exists()
                assert (versioned_path / "metadata.json").exists()
                
                # Check that files were copied
                assert (versioned_path / "test_file_0.txt").exists()
                assert (versioned_path / "test_file_1.txt").exists()


class TestDataVersioning:
    """Test cases for DataVersioning class."""
    
    @pytest.fixture
    def data_versioning(self):
        """Create a DataVersioning instance for testing."""
        return DataVersioning()
    
    def test_initialize_dvc(self, data_versioning):
        """Test DVC initialization."""
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = Path(temp_dir) / "data"
            data_path.mkdir()
            
            data_versioning.initialize_dvc(data_path)
            
            dvc_file = data_path.parent / "data.dvc"
            assert dvc_file.exists()
            
            # Check DVC file content
            content = dvc_file.read_text()
            assert "outs:" in content
            assert "path: data" in content
    
    def test_add_data_version(self, data_versioning):
        """Test adding data version to DVC."""
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = Path(temp_dir) / "data"
            data_path.mkdir()
            
            # This should not raise an exception
            data_versioning.add_data_version(data_path, "Test commit message")
    
    def test_get_data_versions(self, data_versioning):
        """Test getting data versions."""
        versions = data_versioning.get_data_versions()
        
        assert isinstance(versions, list)
        assert len(versions) > 0
        
        # Check version structure
        version = versions[0]
        assert 'version' in version
        assert 'commit_hash' in version
        assert 'created_at' in version
        assert 'file_count' in version


class TestDataValidationResult:
    """Test cases for DataValidationResult dataclass."""
    
    def test_data_validation_result_creation(self):
        """Test DataValidationResult creation."""
        metadata = {'test': 'data'}
        result = DataValidationResult(
            is_valid=True,
            errors=[],
            warnings=['test warning'],
            metadata=metadata
        )
        
        assert result.is_valid is True
        assert len(result.errors) == 0
        assert len(result.warnings) == 1
        assert result.metadata == metadata
    
    def test_data_validation_result_with_errors(self):
        """Test DataValidationResult with errors."""
        result = DataValidationResult(
            is_valid=False,
            errors=['Error 1', 'Error 2'],
            warnings=[],
            metadata={}
        )
        
        assert result.is_valid is False
        assert len(result.errors) == 2
        assert len(result.warnings) == 0


# Integration tests
class TestDataProcessorIntegration:
    """Integration tests for DataProcessor."""
    
    @pytest.fixture
    def sample_data_directory(self):
        """Create a sample data directory for integration testing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = Path(temp_dir) / "data"
            data_path.mkdir()
            
            # Create various test files
            files = [
                ("image1.jpg", b"fake jpg data"),
                ("image2.png", b"fake png data"),
                ("document.txt", b"text file"),
                ("image3.bmp", b"fake bmp data")
            ]
            
            for filename, content in files:
                file_path = data_path / filename
                file_path.write_bytes(content)
            
            yield data_path
    
    def test_end_to_end_data_processing(self, sample_data_directory):
        """Test end-to-end data processing workflow."""
        processor = DataProcessor()
        
        # Step 1: Validate data
        validation_result = processor.validate_data(sample_data_directory)
        
        # Should have 3 valid image files and 1 invalid text file
        assert validation_result.metadata['total_files'] == 3
        assert validation_result.metadata['valid_files'] == 3
        assert validation_result.metadata['invalid_files'] == 0
        
        # Step 2: Create data version
        versioned_path = processor.create_data_version(sample_data_directory, "v1.0")
        
        assert versioned_path.exists()
        assert (versioned_path / "metadata.json").exists()
        
        # Step 3: Verify versioned data
        versioned_files = list(versioned_path.glob("*"))
        assert len(versioned_files) >= 4  # 3 image files + metadata.json


# Performance tests
class TestDataProcessorPerformance:
    """Performance tests for DataProcessor."""
    
    def test_large_image_processing_performance(self):
        """Test performance with large images."""
        processor = DataProcessor()
        
        # Create a large test image
        large_image = np.random.randint(0, 255, (2048, 2048, 3), dtype=np.uint8)
        
        import time
        start_time = time.time()
        
        # Process the large image
        with patch('cv2.resize') as mock_resize:
            mock_resize.return_value = np.random.random((512, 512, 3))
            
            with patch('cv2.cvtColor') as mock_cvtcolor:
                mock_cvtcolor.return_value = np.random.random((512, 512))
                
                with patch('cv2.equalizeHist') as mock_equalize:
                    mock_equalize.return_value = np.random.randint(0, 255, (512, 512))
                    
                    result = processor.preprocess_image(large_image)
        
        processing_time = time.time() - start_time
        
        # Performance assertion (should complete within reasonable time)
        assert processing_time < 5.0  # Should complete within 5 seconds
        assert result is not None
    
    def test_batch_processing_performance(self):
        """Test performance with batch processing."""
        processor = DataProcessor()
        
        # Create multiple test images
        images = [np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8) for _ in range(10)]
        
        import time
        start_time = time.time()
        
        # Process all images
        results = []
        for image in images:
            with patch('cv2.resize') as mock_resize:
                mock_resize.return_value = np.random.random((100, 100, 3))
                
                with patch('cv2.cvtColor') as mock_cvtcolor:
                    mock_cvtcolor.return_value = np.random.random((100, 100))
                    
                    with patch('cv2.equalizeHist') as mock_equalize:
                        mock_equalize.return_value = np.random.randint(0, 255, (100, 100))
                        
                        result = processor.preprocess_image(image)
                        results.append(result)
        
        processing_time = time.time() - start_time
        
        # Performance assertion
        assert processing_time < 10.0  # Should complete within 10 seconds
        assert len(results) == 10
        assert all(result is not None for result in results) 