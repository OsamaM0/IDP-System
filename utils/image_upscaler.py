"""
Image upscaling utilities for the IDP System.

This module provides image upscaling functionality to improve OCR accuracy
on low-resolution images.
"""

import cv2
import numpy as np
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class ImageUpscaler:
    """
    Image upscaling utility class for enhancing low-resolution images.
    
    This class provides various upscaling methods to improve image quality
    before OCR processing.
    """
    
    @staticmethod
    def pixelMapping(image: np.ndarray, scale: int) -> Optional[np.ndarray]:
        """
        Upscale image using pixel mapping (nearest neighbor).
        
        Args:
            image: Input image as numpy array
            scale: Scaling factor
            
        Returns:
            Upscaled image or None if failed
        """
        try:
            height, width = image.shape[:2]
            new_height, new_width = height * scale, width * scale
            
            return cv2.resize(
                image, 
                (new_width, new_height), 
                interpolation=cv2.INTER_NEAREST
            )
        except Exception as e:
            logger.error(f"Pixel mapping upscaling failed: {str(e)}")
            return None
    
    @staticmethod
    def bilinearInterpolation(image: np.ndarray, scale: int) -> Optional[np.ndarray]:
        """
        Upscale image using bilinear interpolation.
        
        Args:
            image: Input image as numpy array
            scale: Scaling factor
            
        Returns:
            Upscaled image or None if failed
        """
        try:
            height, width = image.shape[:2]
            new_height, new_width = height * scale, width * scale
            
            return cv2.resize(
                image, 
                (new_width, new_height), 
                interpolation=cv2.INTER_LINEAR
            )
        except Exception as e:
            logger.error(f"Bilinear interpolation upscaling failed: {str(e)}")
            return None
    
    @staticmethod
    def bicubicInterpolation(image: np.ndarray, scale: int) -> Optional[np.ndarray]:
        """
        Upscale image using bicubic interpolation.
        
        Args:
            image: Input image as numpy array
            scale: Scaling factor
            
        Returns:
            Upscaled image or None if failed
        """
        try:
            height, width = image.shape[:2]
            new_height, new_width = height * scale, width * scale
            
            return cv2.resize(
                image, 
                (new_width, new_height), 
                interpolation=cv2.INTER_CUBIC
            )
        except Exception as e:
            logger.error(f"Bicubic interpolation upscaling failed: {str(e)}")
            return None
    
    @staticmethod
    def lanczosInterpolation(image: np.ndarray, scale: int) -> Optional[np.ndarray]:
        """
        Upscale image using Lanczos interpolation.
        
        Args:
            image: Input image as numpy array
            scale: Scaling factor
            
        Returns:
            Upscaled image or None if failed
        """
        try:
            height, width = image.shape[:2]
            new_height, new_width = height * scale, width * scale
            
            return cv2.resize(
                image, 
                (new_width, new_height), 
                interpolation=cv2.INTER_LANCZOS4
            )
        except Exception as e:
            logger.error(f"Lanczos interpolation upscaling failed: {str(e)}")
            return None
    
    @staticmethod
    def edgePreservingUpscale(image: np.ndarray, scale: int) -> Optional[np.ndarray]:
        """
        Upscale image using edge-preserving interpolation.
        
        Args:
            image: Input image as numpy array
            scale: Scaling factor
            
        Returns:
            Upscaled image or None if failed
        """
        try:
            # First apply basic upscaling
            upscaled = ImageUpscaler.bicubicInterpolation(image, scale)
            if upscaled is None:
                return None
            
            # Apply edge-preserving filter to maintain text clarity
            filtered = cv2.edgePreservingFilter(upscaled, flags=1, sigma_s=50, sigma_r=0.4)
            return filtered
            
        except Exception as e:
            logger.error(f"Edge-preserving upscaling failed: {str(e)}")
            return None
    
    @staticmethod
    def superResolution(image: np.ndarray, scale: int) -> Optional[np.ndarray]:
        """
        Upscale image using OpenCV's super resolution (if available).
        
        Args:
            image: Input image as numpy array
            scale: Scaling factor
            
        Returns:
            Upscaled image or None if failed
        """
        try:
            # Try to use OpenCV's DNN super resolution if available
            # Fallback to bicubic if not available
            try:
                # This might not be available in all OpenCV builds
                sr = cv2.dnn_superres.DnnSuperResImpl_create()
                # You would need to load a pre-trained model here
                # For now, fallback to bicubic
                return ImageUpscaler.bicubicInterpolation(image, scale)
            except AttributeError:
                # DNN super resolution not available, use bicubic
                return ImageUpscaler.bicubicInterpolation(image, scale)
                
        except Exception as e:
            logger.error(f"Super resolution upscaling failed: {str(e)}")
            return None
    
    @staticmethod
    def adaptiveUpscale(image: np.ndarray, scale: int, method: str = 'auto') -> Optional[np.ndarray]:
        """
        Adaptively upscale image based on content analysis.
        
        Args:
            image: Input image as numpy array
            scale: Scaling factor
            method: Upscaling method ('auto', 'text', 'photo', 'mixed')
            
        Returns:
            Upscaled image or None if failed
        """
        try:
            if method == 'auto':
                # Analyze image to determine best method
                # For text documents, use methods that preserve edges
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
                
                # Calculate edge density
                edges = cv2.Canny(gray, 50, 150)
                edge_density = np.sum(edges > 0) / edges.size
                
                if edge_density > 0.1:  # High edge density suggests text
                    method = 'text'
                else:
                    method = 'photo'
            
            if method == 'text':
                # Best for text: preserve sharp edges
                return ImageUpscaler.edgePreservingUpscale(image, scale)
            elif method == 'photo':
                # Best for photos: smooth interpolation
                return ImageUpscaler.bicubicInterpolation(image, scale)
            elif method == 'mixed':
                # Compromise between text and photo
                return ImageUpscaler.lanczosInterpolation(image, scale)
            else:
                return ImageUpscaler.bicubicInterpolation(image, scale)
                
        except Exception as e:
            logger.error(f"Adaptive upscaling failed: {str(e)}")
            return None
    
    @classmethod
    def upscale_with_fallback(cls, image: np.ndarray, scale: int) -> np.ndarray:
        """
        Upscale image with multiple fallback methods.
        
        Args:
            image: Input image as numpy array
            scale: Scaling factor
            
        Returns:
            Upscaled image (guaranteed to return something)
        """
        methods = [
            (cls.edgePreservingUpscale, "edge-preserving"),
            (cls.bicubicInterpolation, "bicubic"),
            (cls.bilinearInterpolation, "bilinear"),
            (cls.pixelMapping, "pixel mapping")
        ]
        
        for method, name in methods:
            try:
                result = method(image, scale)
                if result is not None and result.size > 0:
                    logger.debug(f"Successfully upscaled using {name} method")
                    return result
            except Exception as e:
                logger.debug(f"Upscaling method {name} failed: {str(e)}")
                continue
        
        # Final fallback - just return original image
        logger.warning("All upscaling methods failed, returning original image")
        return image
    
    @staticmethod
    def calculate_optimal_scale(image: np.ndarray, min_dimension: int = 300) -> int:
        """
        Calculate optimal scaling factor based on image dimensions.
        
        Args:
            image: Input image as numpy array
            min_dimension: Minimum dimension for the output image
            
        Returns:
            Optimal scaling factor
        """
        height, width = image.shape[:2]
        current_min_dim = min(height, width)
        
        if current_min_dim >= min_dimension:
            return 1  # No upscaling needed
        
        scale = int(np.ceil(min_dimension / current_min_dim))
        return min(scale, 4)  # Cap at 4x to avoid excessive memory usage
    
    @staticmethod
    def smart_upscale(image: np.ndarray, target_dimension: int = 1000) -> np.ndarray:
        """
        Smart upscaling that automatically determines the best scale and method.
        
        Args:
            image: Input image as numpy array
            target_dimension: Target minimum dimension
            
        Returns:
            Upscaled image
        """
        try:
            # Calculate optimal scale
            scale = ImageUpscaler.calculate_optimal_scale(image, target_dimension)
            
            if scale <= 1:
                return image  # No upscaling needed
            
            # Apply adaptive upscaling
            result = ImageUpscaler.adaptiveUpscale(image, scale, 'auto')
            
            if result is not None:
                return result
            else:
                # Fallback to guaranteed method
                return ImageUpscaler.upscale_with_fallback(image, scale)
                
        except Exception as e:
            logger.error(f"Smart upscaling failed: {str(e)}")
            return image  # Return original if everything fails
