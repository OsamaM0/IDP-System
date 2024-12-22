from core.input.input_source import InputSource
import requests
from io import BytesIO

class UrlInput(InputSource):
    """Input source for loading images from URLs."""

    def __init__(self, url: str):
        self.url = url

    def load_image(self) -> bytes:
        """Downloads the image from the URL and returns its bytes."""
        response = requests.get(self.url, stream=True)
        response.raise_for_status()

        image_data = BytesIO(response.content)
        return image_data.getvalue()