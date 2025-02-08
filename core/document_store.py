import logging
from typing import List, Optional
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

class DocumentStore:
    """Manages the vector store for documents"""
    
    def __init__(self):
        """Initialize the document store"""
        self.vector_store: Optional[Chroma] = None
        self.embedding = OpenAIEmbeddings()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
    
    def add_documents(self, documents: List[Document]) -> None:
        """Add documents to the vector store
        
        Args:
            documents: List of documents to add
        """
        logger.info(f"Processing {len(documents)} document(s)")
        
        # Split documents into chunks
        chunks = self.text_splitter.split_documents(documents)
        logger.info(f"Split into {len(chunks)} chunk(s)")
        
        # Create or update vector store
        if self.vector_store is None:
            logger.info("Creating new vector store")
            self.vector_store = Chroma.from_documents(
                documents=chunks,
                embedding=self.embedding
            )
        else:
            logger.info("Adding to existing vector store")
            self.vector_store.add_documents(chunks)
    
    def search(self, query: str) -> List[Document]:
        """Search for relevant documents
        
        Args:
            query: Search query
            
        Returns:
            List of relevant documents
        """
        if self.vector_store is None:
            return []
        
        return self.vector_store.similarity_search(query)
    
    def clear(self) -> None:
        """Clear the vector store"""
        self.vector_store = None
