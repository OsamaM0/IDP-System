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

        Returns:
            bytes: The scanned image data

        Raises:
            Exception: If scanner interaction fails
        """
        try:
            self.error = None
            # Run the scanning operation in a thread pool to prevent blocking
            loop = asyncio.get_running_loop()
            image_data = await loop.run_in_executor(
                self._executor, 
                self._scan_sync
            )
            return image_data

        except Exception as e:
            self.status = "error"
            self.error = str(e)
            raise

    async def shutdown(self):
        """Cleanup resources."""
        self._executor.shutdown(wait=True)