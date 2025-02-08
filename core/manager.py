import logging
import os
from typing import Optional, Tuple
from .document_store import DocumentStore
from loaders import PDFDocumentLoader, WordDocumentLoader, JiraLoader

logger = logging.getLogger(__name__)

class DocumentManager:
    """Manages document loading and storage operations"""
    
    def __init__(self):
        """Initialize the document manager"""
        self.document_store = DocumentStore()
        self.jira_loader: Optional[JiraLoader] = None
    
    def load_file(self, file_path: str) -> str:
        """Load a document file
        
        Args:
            file_path: Path to the document file
            
        Returns:
            Status message
        """
        try:
            # Determine file type and use appropriate loader
            file_extension = os.path.splitext(file_path)[1].lower()
            
            if file_extension == '.pdf':
                loader = PDFDocumentLoader()
            elif file_extension in ['.docx', '.doc']:
                loader = WordDocumentLoader()
            else:
                return f"Unsupported file type: {file_extension}. Please upload a PDF or Word document."

            # Load and process the document
            documents = loader.load(file_path)
            self.document_store.add_documents(documents)
            
            return "File loaded successfully. You can now ask questions about the document."
        except Exception as e:
            logger.error(f"Error loading file: {str(e)}")
            return f"Error loading file: {str(e)}"
    
    def init_jira(self, server: str, username: str, api_token: str) -> str:
        """Initialize Jira connection
        
        Args:
            server: Jira server URL
            username: Jira username/email
            api_token: Jira API token
            
        Returns:
            Status message
        """
        try:
            self.jira_loader = JiraLoader(server, username, api_token)
            return "Successfully connected to Jira"
        except Exception as e:
            logger.error(f"Failed to connect to Jira: {str(e)}")
            return f"Failed to connect to Jira: {str(e)}"
    
    def load_jira_issue(self, issue_key: str) -> str:
        """Load a Jira issue
        
        Args:
            issue_key: Jira issue key
            
        Returns:
            Status message
        """
        if not self.jira_loader:
            return "Please connect to Jira first"
            
        try:
            documents = self.jira_loader.load(issue_key)
            self.document_store.add_documents(documents)
            return f"Successfully loaded Jira issue {issue_key}"
        except Exception as e:
            logger.error(f"Error loading Jira issue: {str(e)}")
            return f"Error loading Jira issue: {str(e)}"
    
    def clear_state(self) -> Tuple[None, str]:
        """Clear the document store
        
        Returns:
            Tuple of (None for file input, status message)
        """
        self.document_store.clear()
        return None, "State cleared. Ready for new documents."
    
    def search_documents(self, query: str) -> list:
        """Search for documents relevant to the query
        
        Args:
            query: Search query
            
        Returns:
            List of relevant documents
        """
        return self.document_store.search(query)
