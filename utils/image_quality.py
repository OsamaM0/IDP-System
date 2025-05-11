import numpy as np
import cv2
import math
from typing import Dict, Any, Tuple, Optional, List
from utils.logging_utils import logger, exception_handler

class ImageQualityAnalyzer:
    """
    Advanced image quality analysis for document processing.
    
    This class provides methods to analyze various aspects of image quality
    that are critical for OCR accuracy, including blur detection, contrast
    measurement, brightness analysis, and more.
    """
    
    # Quality thresholds for different document types
    QUALITY_THRESHOLDS = {
        "passport": {
            "blur_score": 150,
            "contrast": 40,
            "brightness_range": (90, 190),
            "min_resolution": 1000 * 700
        },
        "id_card": {
            "blur_score": 100,
            "contrast": 30,
            "brightness_range": (80, 200),
            "min_resolution": 800 * 600
        },
        "document": {
            "blur_score": 120,
            "contrast": 35,
            "brightness_range": (85, 195),
            "min_resolution": 900 * 650
        }
    }
    
    # Default thresholds
    DEFAULT_THRESHOLDS = {
        "blur_score": 80,
        "contrast": 25,
        "brightness_range": (70, 220),
        "min_resolution": 600 * 400
    }
    
    @staticmethod
    @exception_handler
    def analyze_image(image: np.ndarray) -> Dict[str, Any]:
        """
        Perform comprehensive quality analysis on an image.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Dictionary containing quality metrics
        """
        if image is None or image.size == 0:
            raise ValueError("Invalid or empty image provided")
            
        # Get basic image properties
        height, width = image.shape[:2]
        channels = 1 if len(image.shape) == 2 else image.shape[2]
        
        # Convert to grayscale for certain metrics if needed
        gray = image if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Compute quality metrics
        blur_score = ImageQualityAnalyzer.detect_blur(gray)
        contrast = ImageQualityAnalyzer.measure_contrast(gray)
        brightness = ImageQualityAnalyzer.measure_brightness(gray)
        noise_level = ImageQualityAnalyzer.estimate_noise(gray)
        has_shadows = ImageQualityAnalyzer.detect_shadows(gray)
        skew_angle = ImageQualityAnalyzer.detect_skew(gray)
        illumination_uniformity = ImageQualityAnalyzer.measure_illumination_uniformity(gray)
        
        # Compile metrics into a dictionary
        metrics = {
            "width": width,
            "height": height,
            "resolution": width * height,
            "aspect_ratio": width / height if height > 0 else 0,
            "channels": channels,
            "blur_score": blur_score,      # Higher is sharper
            "contrast": contrast,          # Percentage
            "brightness": brightness,      # Mean value (0-255)
            "noise_level": noise_level,
            "has_shadows": has_shadows,
            "skew_angle": skew_angle,      # In degrees
            "illumination_uniformity": illumination_uniformity,  # Percentage
            "estimated_dpi": ImageQualityAnalyzer.estimate_dpi(width, height)
        }
        
        # Add qualitative assessments
        metrics.update({
            "is_blurry": blur_score < 100,
            "is_low_contrast": contrast < 30,
            "is_dark": brightness < 80,
            "is_bright": brightness > 200,
            "is_skewed": abs(skew_angle) > 1.0,
            "is_noisy": noise_level > 10,
            "has_uneven_lighting": illumination_uniformity < 70
        })
        
        # Calculate an overall quality score (0-100)
        quality_score = ImageQualityAnalyzer.calculate_quality_score(metrics)
        metrics["quality_score"] = quality_score
        
        return metrics
    
    @staticmethod
    def detect_blur(image: np.ndarray) -> float:
        """
        Detect the amount of blur in an image using Laplacian variance.
        
        Args:
            image: Grayscale image
            
        Returns:
            Blur score (higher means less blurry)
        """
        # Apply Laplacian operator to detect edges
        laplacian = cv2.Laplacian(image, cv2.CV_64F)
        
        # Calculate variance of the edges (lower variance indicates more blur)
        score = laplacian.var()
        
        return score
    
    @staticmethod
    def measure_contrast(image: np.ndarray) -> float:
        """
        Measure image contrast as percentage.
        
        Args:
            image: Grayscale image
            
        Returns:
            Contrast percentage (0-100)
        """
        # Calculate histogram
        hist = cv2.calcHist([image], [0], None, [256], [0, 256])
        hist = hist.flatten() / hist.sum()
        
        # Find non-zero histogram values
        non_zero_vals = hist[hist > 0]
        
        if len(non_zero_vals) <= 1:
            return 0
        
        # Calculate contrast as the range of pixel values with significant frequency
        p5 = np.percentile(image, 5)
        p95 = np.percentile(image, 95)
        
        # Convert to percentage (0-100 scale)
        contrast = 100 * ((p95 - p5) / 255.0)
        
        return contrast
    
    @staticmethod
    def measure_brightness(image: np.ndarray) -> float:
        """
        Calculate the average brightness of an image.
        
        Args:
            image: Grayscale image
            
        Returns:
            Mean brightness value (0-255)
        """
        return float(np.mean(image))
    
    @staticmethod
    def estimate_noise(image: np.ndarray) -> float:
        """
        Estimate the noise level in an image.
        
        Args:
            image: Grayscale image
            
        Returns:
            Estimated noise level (higher means more noise)
        """
        # Apply median filter to remove noise
        filtered_image = cv2.medianBlur(image, 3)
        
        # Calculate the difference between original and filtered image
        diff = cv2.absdiff(image, filtered_image)
        
        # Noise level is the mean of the difference
        noise_level = float(np.mean(diff))
        
        return noise_level
    
    @staticmethod
    def detect_shadows(image: np.ndarray) -> bool:
        """
        Detect if an image contains significant shadows.
        
        Args:
            image: Grayscale image
            
        Returns:
            True if shadows detected, False otherwise
        """
        # Adaptive thresholding to highlight shadow boundaries
        thresh = cv2.adaptiveThreshold(
            image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        # Calculate histogram
        hist = cv2.calcHist([image], [0], None, [256], [0, 256])
        hist = hist.flatten() / hist.sum()
        
        # Check if the histogram has significant dark areas 
        # with a sharp transition to lighter areas
        shadow_indicator = np.sum(hist[:50]) > 0.2 and np.sum(hist[200:]) > 0.2
        
        return shadow_indicator
    
    @staticmethod
    def detect_skew(image: np.ndarray) -> float:
        """
        Detect the skew angle of a document image.
        
        Args:
            image: Grayscale image
            
        Returns:
            Skew angle in degrees (0 if no skew detected)
        """
        try:
            # Threshold the image to get a binary image
            _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
            
            # Apply morphology to enhance lines
            kernel = np.ones((5, 5), np.uint8)
            dilated = cv2.dilate(thresh, kernel, iterations=1)
            
            # Detect lines
            lines = cv2.HoughLinesP(
                dilated, 1, np.pi/180, 
                threshold=100, minLineLength=100, maxLineGap=10
            )
            
            # If no lines detected, return 0
            if lines is None or len(lines) == 0:
                return 0.0
                
            # Calculate angles of the lines
            angles = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if x2 - x1 != 0:  # Avoid division by zero
                    angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                    # Filter angles likely to represent horizontal text lines
                    if abs(angle) < 45:
                        angles.append(angle)
            
            # Return median angle if angles were found
            if angles:
                return float(np.median(angles))
            return 0.0
            
        except Exception:
            # Fallback to 0 if detection fails
            return 0.0
            
    @staticmethod
    def measure_illumination_uniformity(image: np.ndarray) -> float:
        """
        Measure how uniform the lighting is across the image.
        
        Args:
            image: Grayscale image
            
        Returns:
            Uniformity score as percentage (higher is more uniform)
        """
        # Divide the image into blocks and calculate mean brightness in each block
        h, w = image.shape
        block_size = min(h // 4, w // 4, 100)  # Use at least 4×4 grid
        
        if block_size < 10:  # If image is too small, use the whole image
            return 100.0
            
        blocks_h = h // block_size
        blocks_w = w // block_size
        
        block_means = []
        for i in range(blocks_h):
            for j in range(blocks_w):
                # Extract block
                block = image[i*block_size:(i+1)*block_size, j*block_size:(j+1)*block_size]
                block_means.append(np.mean(block))
        
        if not block_means:
            return 100.0  # Perfect uniformity if no blocks
            
        # Calculate coefficient of variation (standard deviation / mean)
        mean_brightness = np.mean(block_means)
        std_brightness = np.std(block_means)
        
        if mean_brightness == 0:
            return 0.0  # Avoid division by zero
            
        cv = (std_brightness / mean_brightness) * 100
        
        # Convert to uniformity (100% - coefficient of variation)
        # Cap at 100% and floor at 0%
        uniformity = max(0, min(100, 100 - cv))
        
        return uniformity
    
    @staticmethod
    def estimate_dpi(width: int, height: int) -> int:
        """
        Estimate the DPI of an image based on size.
        
        Args:
            width: Image width in pixels
            height: Image height in pixels
            
        Returns:
            Estimated DPI
        """
        # Estimate based on typical document sizes
        # A4 paper is approximately 8.27 × 11.69 inches
        # ID card is approximately 3.37 × 2.13 inches
        
        # Assume the larger dimension corresponds to the longer side of common documents
        larger_dimension = max(width, height)
        
        if larger_dimension > 4000:
            return 600  # Very high-res scan or photo
        elif larger_dimension > 2500:
            return 400  # High-res scan
        elif larger_dimension > 1800:
            return 300  # Standard scan
        elif larger_dimension > 1200:
            return 200  # Low-res scan
        else:
            return 100  # Very low-res or thumbnail
    
    @staticmethod
    def calculate_quality_score(metrics: Dict[str, Any]) -> float:
        """
        Calculate an overall quality score based on multiple metrics.
        
        Args:
            metrics: Dictionary of image quality metrics
            
        Returns:
            Quality score from 0 to 100
        """
        score = 100.0
        
        # Penalize for blur
        blur_score = metrics.get("blur_score", 0)
        if blur_score < 50:
            score -= 30
        elif blur_score < 100:
            score -= 15
        elif blur_score < 150:
            score -= 5
            
        # Penalize for poor contrast
        contrast = metrics.get("contrast", 0)
        if contrast < 20:
            score -= 25
        elif contrast < 35:
            score -= 10
        elif contrast < 50:
            score -= 5
            
        # Penalize for extreme brightness
        brightness = metrics.get("brightness", 0)
        if brightness < 40 or brightness > 230:
            score -= 20
        elif brightness < 70 or brightness > 210:
            score -= 10
        elif brightness < 90 or brightness > 190:
            score -= 5
            
        # Penalize for noise
        noise = metrics.get("noise_level", 0)
        if noise > 15:
            score -= 15
        elif noise > 8:
            score -= 7
        elif noise > 5:
            score -= 3
            
        # Penalize for shadows
        if metrics.get("has_shadows", False):
            score -= 12
            
        # Penalize for skew
        skew_angle = abs(metrics.get("skew_angle", 0))
        if skew_angle > 5:
            score -= 15
        elif skew_angle > 3:
            score -= 10
        elif skew_angle > 1:
            score -= 5
            
        # Penalize for uneven illumination
        illumination_uniformity = metrics.get("illumination_uniformity", 100)
        if illumination_uniformity < 50:
            score -= 15
        elif illumination_uniformity < 70:
            score -= 10
        elif illumination_uniformity < 85:
            score -= 5
            
        # Penalize for very low resolution
        resolution = metrics.get("resolution", 0)
        if resolution < 300000:  # Roughly 500x600
            score -= 20
        elif resolution < 600000:  # Roughly 800x750
            score -= 10
            
        # Ensure score is between 0 and 100
        return max(0, min(100, score))
    
    @staticmethod
    def get_enhancement_recommendations(metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate recommendations for image enhancement based on quality analysis.
        
        Args:
            metrics: Dictionary of image quality metrics
            
        Returns:
            Dictionary of recommended enhancement operations
        """
        recommendations = {}
        
        # Blur correction
        if metrics.get("blur_score", 0) < 100:
            recommendations["sharpen"] = {
                "priority": "high" if metrics.get("blur_score", 0) < 50 else "medium",
                "params": {
                    "strength": 1.5 if metrics.get("blur_score", 0) < 50 else 1.2
                }
            }
            
        # Contrast enhancement
        if metrics.get("contrast", 0) < 30:
            recommendations["enhance_contrast"] = {
                "priority": "high" if metrics.get("contrast", 0) < 20 else "medium",
                "method": "clahe" if metrics.get("contrast", 0) < 15 else "histogram",
                "params": {
                    "clip_limit": 3.0 if metrics.get("contrast", 0) < 15 else 2.0,
                    "tile_grid_size": (8, 8)
                }
            }
            
        # Brightness adjustment
        brightness = metrics.get("brightness", 0)
        if brightness < 80:
            recommendations["adjust_brightness"] = {
                "priority": "high" if brightness < 50 else "medium",
                "adjustment": "increase",
                "value": min(50, 80 - brightness)
            }
        elif brightness > 200:
            recommendations["adjust_brightness"] = {
                "priority": "medium",
                "adjustment": "decrease",
                "value": min(50, brightness - 200)
            }
            
        # Shadow removal
        if metrics.get("has_shadows", False):
            recommendations["remove_shadows"] = {
                "priority": "medium",
                "params": {
                    "strength": 1.0
                }
            }
            
        # Deskew
        skew_angle = metrics.get("skew_angle", 0)
        if abs(skew_angle) > 1.0:
            recommendations["deskew"] = {
                "priority": "high" if abs(skew_angle) > 3 else "medium",
                "angle": skew_angle
            }
            
        # Noise removal
        if metrics.get("noise_level", 0) > 8:
            recommendations["denoise"] = {
                "priority": "medium" if metrics.get("noise_level", 0) > 15 else "low",
                "params": {
                    "strength": 10 if metrics.get("noise_level", 0) > 15 else 5
                }
            }
            
        # Upscaling for low resolution
        if metrics.get("resolution", 0) < 500000:
            scale_factor = min(2.0, math.sqrt(1000000 / max(1, metrics.get("resolution", 0))))
            recommendations["upscale"] = {
                "priority": "medium",
                "scale_factor": scale_factor,
                "method": "bicubic"
            }
            
        return recommendations

    @classmethod
    def enhance_image(cls, image: np.ndarray, recommendations: Dict[str, Any]) -> np.ndarray:
        """
        Apply recommended enhancements to improve image quality.
        
        Args:
            image: Input image
            recommendations: Enhancement recommendations from get_enhancement_recommendations()
            
        Returns:
            Enhanced image
        """
        if image is None or image.size == 0:
            raise ValueError("Invalid input image")
            
        if not recommendations:
            return image
            
        # Create a copy of the image to avoid modifying the original
        enhanced = image.copy()
        
        # Sort recommendations by priority
        priority_order = {"high": 0, "medium": 1, "low": 2}
        sorted_enhancements = sorted(
            recommendations.items(), 
            key=lambda x: priority_order.get(x[1].get("priority", "low"), 3)
        )
        
        # Apply enhancements in priority order
        for enhancement, config in sorted_enhancements:
            try:
                if enhancement == "sharpen":
                    # Apply sharpening with configurable strength
                    strength = config.get("params", {}).get("strength", 1.0)
                    kernel = np.array([[-1, -1, -1], 
                                       [-1, 9 + strength, -1], 
                                       [-1, -1, -1]])
                    enhanced = cv2.filter2D(enhanced, -1, kernel)
                    logger.debug(f"Applied sharpening with strength {strength}")
                
                elif enhancement == "enhance_contrast":
                    # Apply contrast enhancement
                    method = config.get("method", "clahe")
                    
                    if method == "clahe":
                        clip_limit = config.get("params", {}).get("clip_limit", 2.0)
                        tile_grid = config.get("params", {}).get("tile_grid_size", (8, 8))
                        
                        if len(enhanced.shape) > 2:  # Color image
                            # Convert to LAB color space for better contrast enhancement
                            lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
                            l, a, b = cv2.split(lab)
                            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
                            cl = clahe.apply(l)
                            enhanced_lab = cv2.merge((cl, a, b))
                            enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
                        else:  # Grayscale image
                            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
                            enhanced = clahe.apply(enhanced)
                    
                    elif method == "histogram":
                        if len(enhanced.shape) > 2:  # Color image
                            # Process each channel separately
                            channels = cv2.split(enhanced)
                            eq_channels = [cv2.equalizeHist(ch) for ch in channels]
                            enhanced = cv2.merge(eq_channels)
                        else:  # Grayscale image
                            enhanced = cv2.equalizeHist(enhanced)
                    
                    logger.debug(f"Applied contrast enhancement using {method}")
                
                elif enhancement == "adjust_brightness":
                    # Adjust brightness
                    adjustment = config.get("adjustment", "increase")
                    value = config.get("value", 30)
                    
                    if adjustment == "increase":
                        if len(enhanced.shape) > 2:  # Color image
                            # Adjust in HSV space for color images
                            hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)
                            h, s, v = cv2.split(hsv)
                            v = cv2.add(v, np.ones(v.shape, dtype="uint8") * value)
                            hsv = cv2.merge([h, s, v])
                            enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                        else:  # Grayscale image
                            enhanced = cv2.add(enhanced, np.ones(enhanced.shape, dtype="uint8") * value)
                    else:  # decrease
                        if len(enhanced.shape) > 2:  # Color image
                            hsv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2HSV)
                            h, s, v = cv2.split(hsv)
                            v = cv2.subtract(v, np.ones(v.shape, dtype="uint8") * value)
                            hsv = cv2.merge([h, s, v])
                            enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
                        else:  # Grayscale image
                            enhanced = cv2.subtract(enhanced, np.ones(enhanced.shape, dtype="uint8") * value)
                    
                    logger.debug(f"Adjusted brightness: {adjustment} by {value}")
                
                elif enhancement == "remove_shadows":
                    # Remove shadows with illumination correction
                    if len(enhanced.shape) > 2:  # Color image
                        # Convert to LAB
                        lab = cv2.cvtColor(enhanced, cv2.COLOR_BGR2LAB)
                        l, a, b = cv2.split(lab)
                        
                        # Apply bilateral filter to L channel to preserve edges
                        l_filtered = cv2.bilateralFilter(l, 9, 75, 75)
                        
                        # Apply CLAHE
                        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                        l_filtered = clahe.apply(l_filtered)
                        
                        # Merge and convert back
                        enhanced_lab = cv2.merge([l_filtered, a, b])
                        enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
                    else:  # Grayscale image
                        # Apply bilateral filter to preserve edges
                        filtered = cv2.bilateralFilter(enhanced, 9, 75, 75)
                        # Apply CLAHE
                        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                        enhanced = clahe.apply(filtered)
                    
                    logger.debug("Applied shadow removal")
                
                elif enhancement == "deskew":
                    # Deskew the image based on detected angle
                    angle = config.get("angle", 0)
                    
                    if abs(angle) > 0.5:  # Only deskew if angle is significant
                        h, w = enhanced.shape[:2]
                        center = (w / 2, h / 2)
                        
                        # Create rotation matrix with the center, angle, and scale
                        M = cv2.getRotationMatrix2D(center, angle, 1)
                        
                        # Apply rotation
                        enhanced = cv2.warpAffine(enhanced, M, (w, h), 
                                                 flags=cv2.INTER_CUBIC,
                                                 borderMode=cv2.BORDER_REPLICATE)
                        logger.debug(f"Applied deskew with angle {angle}")
                
                elif enhancement == "denoise":
                    # Apply denoising
                    strength = config.get("params", {}).get("strength", 10)
                    
                    if len(enhanced.shape) > 2:  # Color image
                        # Use fastNlMeansDenoisingColored for color images
                        enhanced = cv2.fastNlMeansDenoisingColored(enhanced, None, strength, strength, 7, 21)
                    else:  # Grayscale image
                        enhanced = cv2.fastNlMeansDenoising(enhanced, None, strength, 7, 21)
                    
                    logger.debug(f"Applied denoising with strength {strength}")
                
                elif enhancement == "upscale":
                    # Upscale the image
                    scale_factor = config.get("scale_factor", 1.5)
                    method = config.get("method", "bicubic")
                    
                    h, w = enhanced.shape[:2]
                    new_h, new_w = int(h * scale_factor), int(w * scale_factor)
                    
                    if method == "bicubic":
                        enhanced = cv2.resize(enhanced, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
                    elif method == "lanczos":
                        enhanced = cv2.resize(enhanced, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
                    else:  # Default to bilinear
                        enhanced = cv2.resize(enhanced, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                    
                    logger.debug(f"Upscaled image by factor {scale_factor} using {method} method")
                    
            except Exception as e:
                logger.warning(f"Failed to apply {enhancement}: {str(e)}")
        
        # Final enhancements for OCR readability
        try:
            # Ensure uint8 data type and correct range
            enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
            
            # Apply a very mild median filter to reduce remaining noise while preserving edges
            if len(enhanced.shape) > 2:  # Color image
                enhanced = cv2.medianBlur(enhanced, 3)
        except Exception as e:
            logger.warning(f"Failed to apply final processing: {str(e)}")
        
        return enhanced
