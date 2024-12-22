from core.input.input_source import InputSource
import twain  # Assuming you have the twain library installed
import os

class ScannerInput(InputSource):
    """Input source for loading images from scanners."""

    def __init__(self):
        pass

    def load_image(self) -> bytes:
        """
        Captures an image from the scanner and returns its bytes.

        Raises an exception if scanner interaction fails.
        """
        try:
            # Adjust frame size (in inches) based on your scanner's default paper size
            result = twain.acquire(
                # Set output path to a temporary file or handle later
                outpath="temp.jpg",
                dpi=300,
                frame=(0, 0, 8.17551, 11.45438),  # Adjust for paper size (A4)
                pixel_type="bw",
                parent_window=None,  # No window for non-GUI applications
                # show_ui=True,  # Show the scanner's user interface
            )

            if not result:
                raise Exception("Failed to capture image from scanner")

            # Read the captured image from the temporary file (replace with actual logic)
            with open("temp.jpg", "rb") as f:
                image_data = f.read()

            # Clean up the temporary file (replace with actual logic)
            os.remove("temp.jpg")

            return image_data

        except Exception as e:
            raise Exception(f"Error during scanner interaction: {e}")