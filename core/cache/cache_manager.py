"""
Cache manager for the IDP system.
"""
import time
import hashlib
import pickle
import os
from typing import Any, Dict, Optional, Callable, TypeVar, cast
from functools import wraps
from config.config import get_settings
from utils.logging_utils import logger

T = TypeVar('T')

class CacheManager:
    """
    Cache manager for expensive operations.
    """
    def __init__(self):
        self.settings = get_settings()
        self.enable_caching = self.settings.ENABLE_CACHING
        self.cache_ttl = self.settings.CACHE_TTL
        self.cache_dir = os.path.join(os.getcwd(), "cache")
        
        # Create cache directory if it doesn't exist
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
        
        # In-memory cache
        self.memory_cache: Dict[str, Dict[str, Any]] = {}
        
        logger.debug(f"Cache manager initialized with TTL {self.cache_ttl}s, caching {'enabled' if self.enable_caching else 'disabled'}")
    
    def _get_cache_key(self, *args, **kwargs) -> str:
        """
        Generate a unique cache key based on function arguments.
        
        Args:
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            str: A unique cache key
        """
        # Create a string representation of args and kwargs
        key_parts = [str(arg) for arg in args]
        key_parts.extend(f"{k}={v}" for k, v in sorted(kwargs.items()))
        key_str = "|".join(key_parts)
        
        # Hash the string to get a fixed-length key
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get(self, cache_key: str) -> Optional[Any]:
        """
        Get an item from the cache.
        
        Args:
            cache_key: The cache key
            
        Returns:
            The cached item or None if not found or expired
        """
        if not self.enable_caching:
            return None
        
        # Check memory cache first
        cache_data = self.memory_cache.get(cache_key)
        if cache_data:
            expiry_time = cache_data.get("expiry")
            if expiry_time and time.time() < expiry_time:
                logger.debug(f"Cache hit (memory): {cache_key}")
                return cache_data.get("data")
        
        # Check disk cache
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.cache")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, "rb") as f:
                    try:
                        cache_data = pickle.load(f)
                    except EOFError:
                        logger.warning(f"Cache file {cache_file} is empty. Returning None.")
                        return None
                    
                expiry_time = cache_data.get("expiry")
                if expiry_time and time.time() < expiry_time:
                    # Update memory cache
                    self.memory_cache[cache_key] = cache_data
                    logger.debug(f"Cache hit (disk): {cache_key}")
                    return cache_data.get("data")
                else:
                    # Remove expired cache file
                    os.remove(cache_file)
            except (pickle.PickleError, IOError):
                # If any error occurs, ignore the cache
                pass
        
        logger.debug(f"Cache miss: {cache_key}")
        return None
    
    def set(self, cache_key: str, data: Any, ttl: Optional[int] = None) -> None:
        """
        Set an item in the cache.
        
        Args:
            cache_key: The cache key
            data: The data to cache
            ttl: Time to live in seconds (overrides default TTL)
        """
        if not self.enable_caching:
            return
        
        expiry_time = time.time() + (ttl or self.cache_ttl)
        
        # Update memory cache
        cache_data = {
            "data": data,
            "expiry": expiry_time
        }
        self.memory_cache[cache_key] = cache_data
        
        # Update disk cache
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.cache")
        try:
            with open(cache_file, "wb") as f:
                pickle.dump(cache_data, f)
        except (pickle.PickleError, TypeError, IOError) as e:
            logger.warning(f"Failed to write cache to disk for key '{cache_key}': {str(e)}")
            # Remove from memory cache to prevent inconsistency
            if cache_key in self.memory_cache:
                del self.memory_cache[cache_key]
    
    def invalidate(self, cache_key: str) -> None:
        """
        Invalidate a specific cache entry.
        
        Args:
            cache_key: The cache key to invalidate
        """
        # Remove from memory cache
        if cache_key in self.memory_cache:
            del self.memory_cache[cache_key]
        
        # Remove from disk cache
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.cache")
        if os.path.exists(cache_file):
            try:
                os.remove(cache_file)
            except IOError as e:
                logger.warning(f"Failed to remove cache file: {str(e)}")
    
    def clear_all(self) -> None:
        """Clear all cache entries."""
        # Clear memory cache
        self.memory_cache = {}
        
        # Clear disk cache
        for file_name in os.listdir(self.cache_dir):
            if file_name.endswith(".cache"):
                try:
                    os.remove(os.path.join(self.cache_dir, file_name))
                except IOError:
                    pass


# Global cache manager instance
cache_manager = CacheManager()


def cached(ttl: Optional[int] = None):
    """
    Decorator to cache function results.
    
    Args:
        ttl: Optional time to live in seconds
        
    Returns:
        The decorated function
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not cache_manager.enable_caching:
                return func(*args, **kwargs)
            
            # Generate cache key
            cache_key = f"{func.__module__}.{func.__name__}:{cache_manager._get_cache_key(*args, **kwargs)}"
            
            # Try to get from cache
            cached_result = cache_manager.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            cache_manager.set(cache_key, result, ttl)
            return result
        
        return cast(Callable[..., T], wrapper)
    
    return decorator
