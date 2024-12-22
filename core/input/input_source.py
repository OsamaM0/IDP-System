from abc import ABC, abstractmethod

class InputSource(ABC):
    """Abstract base class for all input sources."""

    @abstractmethod
    def load_image(self) -> bytes:
        """Loads the image data from the source."""
        pass