import logging
import os
from typing import List, Optional
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import chromadb

logger = logging.getLogger(__name__)

class DocumentStore:
    """Manages the vector store for documents"""
    
    def __init__(self, persist_directory: str = "_db"):
        """Initialize the document store
        
        Args:
            persist_directory: Directory to store the vector database
        """
        self.persist_directory = persist_directory
        os.makedirs(persist_directory, exist_ok=True)
        
        self.embedding = OpenAIEmbeddings()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        
        # Initialize persistent vector store
        client = chromadb.PersistentClient(path=persist_directory)
        self.vector_store = Chroma(
            client=client,
            embedding_function=self.embedding,
            collection_name="documents"
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
        
        # Add documents to vector store
        logger.info("Adding documents to vector store")
        self.vector_store.add_documents(chunks)
    
    def search(self, query: str) -> List[Document]:
        """Search for relevant documents
        
        Args:
            query: Search query
            
        Returns:
            List of relevant documents
        """
        return self.vector_store.similarity_search(query)
    
    def clear(self) -> None:
        """Clear the vector store"""
        logger.info("Clearing vector store")
        self.vector_store._collection.delete(where={})
