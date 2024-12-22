from core.input.input_source import InputSource
from PIL import Image

class FileInput(InputSource):
    """Input source for loading images from local files."""

    def __init__(self, file_path: str):
        self.file_path = file_path

    def load_image(self) -> bytes:
        """Loads the image as bytes from the specified file path."""
        with Image.open(self.file_path) as image:
            return image.convert("RGB").tobytes()