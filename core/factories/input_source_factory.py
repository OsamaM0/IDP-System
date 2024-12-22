from core.input.file_input import FileInput
from core.input.input_source import InputSource
from core.input.url_input import UrlInput
from core.input.scanner_input import ScannerInput
import os

class InputSourceFactory:
    """Factory class to create input source objects based on the input type."""

    @staticmethod
    def create_input_source(input_data: str) -> InputSource:
        """
        Creates an InputSource object based on the type of input data.

        Args:
            input_data: The input data to be processed (e.g., file path, URL).

        Returns:
            An InputSource object or raises an exception if the input type is not supported.
        """

        if input_data.startswith("http"):
            return UrlInput(input_data)
        elif os.path.isfile(input_data):
            return FileInput(input_data)
        elif isinstance(input_data, str):  # Check for scanner input string identifier (optional)
            # You can define a specific string to represent scanner input (e.g., "scanner")
            if input_data == "scanner":
                return ScannerInput()
            else:
                raise ValueError(f"Unsupported input type: {input_data}")
        else:
            raise ValueError(f"Unsupported input type: {input_data}")