from abc import ABC, abstractmethod
from typing import List
from langchain_core.documents import Document

class BaseLoader(ABC):
    """Base class for all document loaders"""
    
    @abstractmethod
    def load(self, source: str) -> List[Document]:
        """Load documents from the source
        
        Args:
            source: Source to load documents from (file path, issue key, etc.)
            
        Returns:
            List of loaded documents
        """
        pass
