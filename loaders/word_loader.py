from typing import List
import logging
from langchain_community.document_loaders import Docx2txtLoader
from langchain_core.documents import Document
from .base_loader import BaseLoader

logger = logging.getLogger(__name__)

class WordDocumentLoader(BaseLoader):
    """Loader for Word documents"""
    
    def load(self, source: str) -> List[Document]:
        """Load documents from a Word file
        
        Args:
            source: Path to the Word file
            
        Returns:
            List of loaded documents
        """
        logger.info(f"Loading Word file: {source}")
        loader = Docx2txtLoader(source)
        return loader.load()
