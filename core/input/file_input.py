from core.input.input_source import InputSource
from PIL import Image
from io import BytesIO

class FileInput(InputSource):
    """Input source for loading images from local files."""

    def __init__(self, file_path: str):
        self.file_path = file_path

    def load_image(self) -> bytes:
        """Loads the image as bytes from the specified file path.
        
        Returns:
            bytes: The raw image data in bytes format.
        
        Raises:
            FileNotFoundError: If the specified file path doesn't exist.
            PIL.UnidentifiedImageError: If the file is not a valid image.
        """
        with Image.open(self.file_path) as image:
            # Convert image to RGB mode if it's not already
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Use BytesIO to get the image data in bytes format
            img_byte_arr = BytesIO()
            image.save(img_byte_arr, format='PNG')
            return img_byte_arr.getvalue()