"""
Logging Utility Module

This module provides centralized logging functionality for the MLOps OCR demo.
Includes structured logging, log rotation, and different log levels.
"""

import logging
import sys
from pathlib import Path
from typing import Optional
import structlog
from src.config.config import config


def setup_logger(
    name: str = "ocr_demo",
    level: str = "INFO",
    log_file: Optional[Path] = None,
    format_string: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
) -> structlog.BoundLogger:
    """
    Set up structured logging for the application.
    
    Args:
        name: Logger name
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional log file path
        format_string: Log format string
    
    Returns:
        Configured structured logger
    """
    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )
    
    # Get the logger
    logger = structlog.get_logger(name)
    
    # Set up standard library logging
    stdlib_logger = logging.getLogger(name)
    stdlib_logger.setLevel(getattr(logging, level.upper()))
    
    # Create formatter
    formatter = logging.Formatter(format_string)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level.upper()))
    console_handler.setFormatter(formatter)
    stdlib_logger.addHandler(console_handler)
    
    # File handler (if log_file is specified)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, level.upper()))
        file_handler.setFormatter(formatter)
        stdlib_logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str = "ocr_demo") -> structlog.BoundLogger:
    """
    Get a logger instance with the specified name.
    
    Args:
        name: Logger name
    
    Returns:
        Structured logger instance
    """
    return structlog.get_logger(name)


# Create default logger
default_logger = setup_logger(
    name="ocr_demo",
    level=config.logging.level,
    log_file=config.logging.log_file,
    format_string=config.logging.format
)


class LoggerMixin:
    """Mixin class to add logging capabilities to any class."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = get_logger(self.__class__.__name__)
    
    def log_info(self, message: str, **kwargs):
        """Log info message with additional context."""
        self.logger.info(message, **kwargs)
    
    def log_warning(self, message: str, **kwargs):
        """Log warning message with additional context."""
        self.logger.warning(message, **kwargs)
    
    def log_error(self, message: str, **kwargs):
        """Log error message with additional context."""
        self.logger.error(message, **kwargs)
    
    def log_debug(self, message: str, **kwargs):
        """Log debug message with additional context."""
        self.logger.debug(message, **kwargs)


def log_function_call(func):
    """Decorator to log function calls with parameters and return values."""
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        logger.info(
            "Function called",
            function_name=func.__name__,
            args=args,
            kwargs=kwargs
        )
        
        try:
            result = func(*args, **kwargs)
            logger.info(
                "Function completed successfully",
                function_name=func.__name__,
                result_type=type(result).__name__
            )
            return result
        except Exception as e:
            logger.error(
                "Function failed",
                function_name=func.__name__,
                error=str(e),
                error_type=type(e).__name__
            )
            raise
    
    return wrapper


def log_performance(func):
    """Decorator to log function performance metrics."""
    import time
    
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.info(
                "Function performance",
                function_name=func.__name__,
                execution_time_seconds=execution_time
            )
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(
                "Function failed",
                function_name=func.__name__,
                execution_time_seconds=execution_time,
                error=str(e)
            )
            raise
    
    return wrapper 