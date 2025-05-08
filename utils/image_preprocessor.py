from PIL import Image
import cv2
import numpy as np
import math 
import PIL
class ImagePreprocessor:
    INTER_NEAREST = cv2.INTER_NEAREST
    INTER_LINEAR = cv2.INTER_LINEAR
    INTER_CUBIC = cv2.INTER_CUBIC
    INTER_AREA = cv2.INTER_AREA
    INTER_LANCZOS4 = cv2.INTER_LANCZOS4
    THRESH_BINARY = cv2.THRESH_BINARY
    THRESH_BINARY_INV = cv2.THRESH_BINARY_INV
    THRESH_OTSU = cv2.THRESH_OTSU
    COLOR_GRAY2RGB = cv2.COLOR_GRAY2RGB
    COLOR_RGB2GRAY = cv2.COLOR_RGB2GRAY
    MORPH_RECT = cv2.MORPH_RECT
    
    @staticmethod
    def noise_removal(image, ksize=5):
        """
        Applies median blur to remove noise.

        Args:
            image (numpy.ndarray): The input image.
            ksize (int): Size of the kernel for median blur. Defaults to 5.

        Returns:
            numpy.ndarray: The image with noise removed.
        """
        return cv2.medianBlur(image, ksize)

    @staticmethod
    def adaptive_thresholding(image, block_size=11, c=2):
        """
        Applies adaptive thresholding to the image.

        Args:
            image (numpy.ndarray): The input image.
            block_size (int): Size of the neighborhood for threshold calculation.
                                Must be odd. Defaults to 11.
            c (int): Constant subtracted from the mean or weighted mean. 
                     Defaults to 2.

        Returns:
            numpy.ndarray: The thresholded image.
        """
        return cv2.adaptiveThreshold(
            image, 
            255, 
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 
            block_size, 
            c
        )

    @staticmethod
    def binarization(image, threshold=128, max_value=255, 
                     threshold_type=cv2.THRESH_BINARY, 
                     inverted=False, otsu=False):
        """
        Applies binary thresholding to the image.

        Args:
            image (numpy.ndarray): The input image.
            threshold (int): Threshold value. Defaults to 128.
            max_value (int): Maximum value for pixels above threshold. 
                            Defaults to 255.
            threshold_type (int): Type of thresholding. 
                                 Defaults to cv2.THRESH_BINARY.
            inverted (bool): Whether to invert the threshold. 
                             Defaults to False.
            otsu (bool): Whether to use Otsu's method for automatic thresholding.
                         Defaults to False.

        Returns:
            tuple: A tuple containing the threshold value and the thresholded image.
        """
        if inverted:
            threshold_type = cv2.THRESH_BINARY_INV
        if otsu:
            threshold_type += cv2.THRESH_OTSU 

        return cv2.threshold(
            image, threshold, max_value, threshold_type
        )[1]

    @staticmethod
    def convert_to_rgb(image):
        """
        Converts the image to RGB format.

        Args:
            image (numpy.ndarray): The input image.

        Returns:
            numpy.ndarray: The image in RGB format.
        """
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    @staticmethod
    def convert_to_grayscale(image: np.ndarray) -> np.ndarray:
        """
        Converts the image to grayscale format.

        Args:
            image (numpy.ndarray): The input image.

        Returns:
            numpy.ndarray: The image in grayscale format.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)            # Step 1: Grayscale (H, W)
        gray_3channel = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)     # Step 2: Expand to (H, W, 3)
        return gray_3channel
    
    @staticmethod
    def denoise(image):
        """Remove noise while preserving edges"""
        # Non-local means denoising
        return cv2.fastNlMeansDenoisingColored(image, None, 3, 3, 7, 31)
    
    @staticmethod
    def blur(image, ksize=(3, 3)):
        """Apply Gaussian blur to the image"""
        return cv2.GaussianBlur(image, ksize, 0)
    
    @staticmethod
    def erode(image, ksize=(3, 3)):
        """Erode the image"""
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, ksize)
        return cv2.erode(image, kernel, iterations=1)
    
    @staticmethod
    def dilate(image, ksize=(3, 3)):
        """Dilate the image"""
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, ksize)
        return cv2.dilate(image, kernel, iterations=1)
    
    @staticmethod
    def resize(cropped_image, scale=1, interpolation=cv2.INTER_CUBIC, image_shape=None):
        """
        Resize the image.
        """
        resized = cv2.resize(cropped_image, image_shape, fx=scale, fy=scale, interpolation=interpolation)
        return resized

    @staticmethod
    def cast_and_scale(cast_type: str, image: np.ndarray) -> np.ndarray:
        """
        Cast the image to the specified type.
        """
        if cast_type == "float32":
            return np.asarray(np.float32(image / 255))
        else:
            raise ValueError(f"Unsupported cast type: {cast_type}")

    @staticmethod
    def reshape(image, image_shape=None):
        """
        Reshape the image to the specified shape.
        """
        if image_shape is not None:
            return np.reshape(image, image_shape)
        return image

    @staticmethod
    def expand_image(image_dim, scale_height=1.2, scale_width=1.0, image_shape=None):
        """
        Expands the bounding box height, width, or both based on the given scales.

        Args:
            image_dim (list): The image dimission [x1, y1, x2, y2].
            scale_height (float): Scale factor for height. Defaults to 1.2.
            scale_width (float): Scale factor for width. Defaults to 1.0.
            image_shape (tuple): Shape of the image (height, width). Defaults to None.

        Returns:
            list: The expanded image dimission [x1, y1, x2, y2].
        """
        x1, y1, x2, y2 = image_dim
        width = x2 - x1
        height = y2 - y1
        center_x = x1 + width // 2
        center_y = y1 + height // 2

        # Calculate new dimensions
        new_height = int(height * scale_height)
        new_width = int(width * scale_width)

        # Calculate new bounding box coordinates
        new_x1 = max(center_x - new_width // 2, 0)
        new_x2 = min(center_x + new_width // 2, image_shape[1] if image_shape else float('inf'))
        new_y1 = max(center_y - new_height // 2, 0)
        new_y2 = min(center_y + new_height // 2, image_shape[0] if image_shape else float('inf'))

        return [new_x1, new_y1, new_x2, new_y2]
    

    @staticmethod
    def expand_image_background(image: np.ndarray, scale_height=1.2, scale_width=1.2, background_color=(255, 255, 255)):
        """
        Expands the image by scaling its height, width, or both, placing it on a white background.

        Args:
            image (PIL.Image): Input image to expand.
            scale_height (float): Scale factor for height. Defaults to 1.2.
            scale_width (float): Scale factor for width. Defaults to 1.0.
            background_color (tuple): RGB color for background. Defaults to white (255, 255, 255).

        Returns:
            PIL.Image: Expanded image with white background.
        """

        # Convert array to PIL Image
        pil_image = Image.fromarray(image)

        # Get original dimensions
        width, height = pil_image.size

        # Calculate new dimensions
        new_width = int(width * scale_width)
        new_height = int(height * scale_height)

        # Extract background color (top-left pixel)
        background_color = tuple(image[0, 0])

        # Create new image with the same background color
        new_image = Image.new('RGB', (new_width, new_height), background_color)

        # Calculate position to paste original image (centered)
        paste_x = (new_width - width) // 2
        paste_y = (new_height - height) // 2

        # Paste the original image onto the new canvas
        new_image.paste(pil_image, (paste_x, paste_y))

        return np.array(new_image)