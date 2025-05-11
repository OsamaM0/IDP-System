"""
Security middleware for the API.
"""
import time
from typing import Dict, List, Callable
from fastapi import Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from config.config import get_settings
from utils.logging_utils import logger

class RateLimiter:
    """
    Rate limiter implementation based on fixed window algorithm.
    """
    def __init__(self, limit_per_minute: int = 100):
        self.limit = limit_per_minute
        self.window_size = 60  # seconds in a minute
        self._requests: Dict[str, List[float]] = {}
    
    def is_allowed(self, key: str) -> bool:
        """
        Check if a request is allowed based on rate limits.
        
        Args:
            key: Identification key (e.g., IP address)
            
        Returns:
            True if the request is allowed, False otherwise
        """
        current_time = time.time()
        
        # Initialize if key not present
        if key not in self._requests:
            self._requests[key] = []
        
        # Clean up old requests
        self._requests[key] = [t for t in self._requests[key] if current_time - t < self.window_size]
        
        # Check if under limit
        if len(self._requests[key]) < self.limit:
            self._requests[key].append(current_time)
            return True
        
        return False


class SecurityMiddleware(BaseHTTPMiddleware):
    """
    Middleware for handling security concerns:
    - Rate limiting
    - Security headers
    - API key validation
    """
    def __init__(
        self, 
        app,
        rate_limit_per_minute: int = 100,
        api_key: str = None
    ):
        super().__init__(app)
        settings = get_settings()
        self.api_key = api_key or settings.API_KEY
        self.rate_limiter = RateLimiter(rate_limit_per_minute or settings.RATE_LIMIT)
        self.rate_limit_enabled = settings.RATE_LIMIT_ENABLED
        self.exempt_paths = [
            "/docs", 
            "/redoc", 
            "/openapi.json",
            "/docs/oauth2-redirect",
            "/favicon.ico"
        ]
    
    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        """
        Process the request.
        
        Args:
            request: The incoming request
            call_next: The next middleware or route handler
            
        Returns:
            The response
        """
        # Get client IP
        client_ip = request.client.host if request.client else "unknown"
        
        # Check if the path should be exempt from security checks
        path = request.url.path
        if path in self.exempt_paths or path.startswith("/docs/") or path.startswith("/redoc/"):
            return await call_next(request)
        
        # API key validation for protected routes
        if self._is_protected_route(request.url.path):
            if not self._validate_api_key(request):
                logger.warning(f"Unauthorized API access attempt from {client_ip}: Invalid API key")
                return JSONResponse(
                    status_code=401,
                    content={"detail": "Invalid or missing API key"}
                )
        
        # Rate limiting
        if self.rate_limit_enabled and not self.rate_limiter.is_allowed(client_ip):
            logger.warning(f"Rate limit exceeded for {client_ip}")
            return JSONResponse(
                status_code=429,
                content={"detail": "Too many requests"}
            )
        
        # Continue with the request
        response = await call_next(request)
        
        # Add security headers
        self._add_security_headers(response)
        
        return response
    
    def _is_protected_route(self, path: str) -> bool:
        """
        Check if a route requires API key authentication.
        
        Args:
            path: The request path
            
        Returns:
            True if the route is protected, False otherwise
        """
        # Skip API key check for docs and static files
        if path.startswith("/docs") or path.startswith("/redoc") or path.startswith("/static"):
            return False
        
        # Skip for OpenAPI schema
        if path == "/openapi.json":
            return False
        
        # Protect all /v1/ routes (adjust as needed)
        if path.startswith("/v1/"):
            return True
        
        # Protect specific endpoints
        protected_paths = [
            "/idp/process-idp",
            "/ocr/ocr",
            "/document_types/verify-document",
            "/roi/roi",
        ]
        
        return any(path.startswith(p) for p in protected_paths)
    
    def _validate_api_key(self, request: Request) -> bool:
        """
        Validate the API key from the request.
        
        Args:
            request: The incoming request
            
        Returns:
            True if the API key is valid, False otherwise
        """
        # Skip validation if no API key is configured
        if not self.api_key:
            return True
        
        # Check Authorization header
        auth_header = request.headers.get("Authorization")
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header.replace("Bearer ", "")
            return token == self.api_key
        
        # Check X-API-Key header
        api_key_header = request.headers.get("X-API-Key")
        if api_key_header and api_key_header == self.api_key:
            return True
        
        # Check query parameter
        api_key_param = request.query_params.get("api_key")
        if api_key_param and api_key_param == self.api_key:
            return True
        
        return False
    
    def _add_security_headers(self, response: Response) -> None:
        """
        Add security headers to the response.
        
        Args:
            response: The response object
        """
        # Set security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Content-Security-Policy"] = "default-src 'self'"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
