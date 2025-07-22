"""
Logging utilities for the IDP System.

This module provides logging configuration, decorators, and utilities
for consistent logging across the application.
"""

import logging
import sys
import time
import functools
from typing import Any, Callable, Optional
import traceback
from pathlib import Path


# Configure logging
def setup_logger(
    name: str = "idp_system",
    level: int = logging.INFO,
    log_file: Optional[str] = None
) -> logging.Logger:
    """
    Setup and configure logger.
    
    Args:
        name: Logger name
        level: Logging level
        log_file: Optional log file path
        
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    
    # Avoid adding handlers multiple times
    if logger.handlers:
        return logger
    
    logger.setLevel(level)
    
    # Create formatter
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler if specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


# Create default logger
logger = setup_logger(
    name="idp_system",
    level=logging.INFO,
    log_file="idp_system.log"
)


def log_execution_time(func: Callable) -> Callable:
    """
    Decorator to log function execution time.
    
    Args:
        func: Function to decorate
        
    Returns:
        Decorated function
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        
        # Log function start
        logger.debug(f"Starting execution of {func.__name__}")
        
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            # Log successful completion
            logger.info(
                f"Function {func.__name__} completed successfully in {execution_time:.3f} seconds"
            )
            
            return result
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(
                f"Function {func.__name__} failed after {execution_time:.3f} seconds: {str(e)}"
            )
            raise
    
    return wrapper


def exception_handler(func: Callable) -> Callable:
    """
    Decorator to handle and log exceptions.
    
    Args:
        func: Function to decorate
        
    Returns:
        Decorated function
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            # Log the full traceback
            logger.error(
                f"Exception in {func.__name__}: {str(e)}\n"
                f"Traceback: {traceback.format_exc()}"
            )
            
            # Re-raise the exception
            raise
    
    return wrapper


def log_function_call(func: Callable) -> Callable:
    """
    Decorator to log function calls with arguments.
    
    Args:
        func: Function to decorate
        
    Returns:
        Decorated function
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # Create argument string (limit length for readability)
        args_str = str(args)[:100] + "..." if len(str(args)) > 100 else str(args)
        kwargs_str = str(kwargs)[:100] + "..." if len(str(kwargs)) > 100 else str(kwargs)
        
        logger.debug(
            f"Calling {func.__name__} with args: {args_str}, kwargs: {kwargs_str}"
        )
        
        return func(*args, **kwargs)
    
    return wrapper


# Convenience functions for different log levels
def debug(message: str, *args, **kwargs) -> None:
    """Log debug message."""
    logger.debug(message, *args, **kwargs)


def info(message: str, *args, **kwargs) -> None:
    """Log info message."""
    logger.info(message, *args, **kwargs)


def warning(message: str, *args, **kwargs) -> None:
    """Log warning message."""
    logger.warning(message, *args, **kwargs)


def error(message: str, *args, **kwargs) -> None:
    """Log error message."""
    logger.error(message, *args, **kwargs)


def critical(message: str, *args, **kwargs) -> None:
    """Log critical message."""
    logger.critical(message, *args, **kwargs)
