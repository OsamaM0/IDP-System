"""
Logging utilities for the IDP system.
"""
import logging
import json
import sys
import time
from functools import wraps
from typing import Any, Callable, Dict, Optional
import traceback
from config.config import get_settings

# Create logger
logger = logging.getLogger("idp_system")


class StructuredLogFormatter(logging.Formatter):
    """
    JSON formatter for structured logging.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def format(self, record):
        log_record = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno,
        }
        
        # Add exception info if present
        if record.exc_info:
            log_record["exception"] = self.formatException(record.exc_info)
        
        # Add extra fields from the record
        for key, value in record.__dict__.items():
            if key not in ["args", "asctime", "created", "exc_info", "exc_text", "filename",
                          "funcName", "id", "levelname", "levelno", "lineno", "module",
                          "msecs", "message", "msg", "name", "pathname", "process",
                          "processName", "relativeCreated", "stack_info", "thread", "threadName"]:
                log_record[key] = value
        
        return json.dumps(log_record)


def setup_logging():
    """
    Set up structured logging for the application.
    """
    settings = get_settings()
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(StructuredLogFormatter())
    
    # Create file handler for important logs
    file_handler = logging.FileHandler("idp_system.log")
    file_handler.setFormatter(StructuredLogFormatter())
    file_handler.setLevel(logging.WARNING)  # Only warnings and above go to file
    
    # Configure the logger
    logger.setLevel(getattr(logging, settings.LOG_LEVEL))
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    logger.debug("Logging setup complete")
    return logger


def log_execution_time(func: Callable) -> Callable:
    """
    Decorator to log function execution time.
    
    Args:
        func: The function to be decorated
        
    Returns:
        The decorated function
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Get function name and module
        func_name = func.__name__
        module_name = func.__module__
        
        logger.debug(f"Function {module_name}.{func_name} executed in {execution_time:.4f} seconds")
        
        return result
    return wrapper


def exception_handler(func: Callable) -> Callable:
    """
    Decorator to handle and log exceptions.
    
    Args:
        func: The function to be decorated
        
    Returns:
        The decorated function
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            # Get function details
            func_name = func.__name__
            module_name = func.__module__
            
            # Get full traceback
            tb = traceback.format_exc()
            
            # Log the exception
            logger.error(
                f"Exception in {module_name}.{func_name}: {str(e)}",
                extra={
                    "exception_type": type(e).__name__,
                    "traceback": tb
                }
            )
            
            # Re-raise the exception
            raise
    
    return wrapper


# Initialize logging on module import
setup_logging()
