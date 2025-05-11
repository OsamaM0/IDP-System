from core.input.input_source import InputSource
import twain
import os
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Optional
import tempfile
import logging

class ScannerInput(InputSource):
    """Input source for loading images from scanners with async support."""

    def __init__(self):
        self.status: str = "idle"
        self.error: Optional[str] = None
        self._executor = ThreadPoolExecutor(max_workers=1)
        self.logger = logging.getLogger(__name__)

    async def get_status(self) -> dict:
        """Get current scanner status."""
        return {
            "status": self.status,
            "error": self.error
        }
    
    def _scan_sync(self) -> bytes:
        """Synchronous scanning operation."""
        try:
            self.status = "initializing"
            # Create a temporary file with .jpg extension
            with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
                temp_path = temp_file.name

            self.status = "scanning"
            result = twain.acquire(
                temp_path,
                "CS 2553ci",
                dpi=300,
                frame=(0, 0, 8.17551, 11.45438),  # A4 size
                pixel_type="color",
                parent_window=None,
            )
    
            if not result:
                raise Exception("Scanner failed to capture image")

            self.status = "processing"
            with open(temp_path, "rb") as f:
                image_data = f.read()

            # Clean up
            os.remove(temp_path)
            
            self.status = "completed"
            return image_data

        except Exception as e:
            self.error = str(e)
            self.logger.error(f"Scanner error: {e}")
            raise

        finally:
            if os.path.exists(temp_path):
                try:
                    os.remove(temp_path)
                except Exception as e:
                    self.logger.error(f"Failed to remove temporary file: {e}")

    async def load_image(self) -> bytes:
        """
        Asynchronously captures an image from the scanner and returns its bytes.
        Raises an exception if no valid image data is received.
        """
        try:
            self.error = None
            loop = asyncio.get_running_loop()
            image_data = await loop.run_in_executor(
                self._executor, 
                self._scan_sync
            )
            if not image_data:
                raise Exception("No image data captured from scanner")
            return image_data

        except Exception as e:
            self.status = "error"
            self.error = str(e)
            self.logger.error(f"Error in load_image: {e}")
            raise

    async def shutdown(self):
        """Cleanup resources."""
        try:
            self._executor.shutdown(wait=True)
            self.logger.info("Scanner executor shutdown successfully.")
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")