from typing import List
import logging
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from .base_loader import BaseLoader

logger = logging.getLogger(__name__)

class PDFDocumentLoader(BaseLoader):
    """Loader for PDF documents"""
    
    def load(self, source: str) -> List[Document]:
        """Load documents from a PDF file
        
        Args:
            source: Path to the PDF file
            
        Returns:
            List of loaded documents
        """
        logger.info(f"Loading PDF file: {source}")
        loader = PyPDFLoader(source)
        return loader.load()
