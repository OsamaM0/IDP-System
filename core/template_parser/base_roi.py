from abc import ABC, abstractmethod
from typing import Any, Dict, List

class BaseParser(ABC):
    @abstractmethod
    def parse(self) -> List[Dict[str, Any]]:
        """
        Extract text from the specified ROI in the image.

        Returns:
            List[Dict[str, Any]]: Extracted text with keys from ROI.
        """
        pass
