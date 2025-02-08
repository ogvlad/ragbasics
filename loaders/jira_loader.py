from typing import List, Optional
import logging
from jira import JIRA
from langchain_core.documents import Document
from .base_loader import BaseLoader

logger = logging.getLogger(__name__)

class JiraLoader(BaseLoader):
    """Loader for Jira issues"""
    
    def __init__(self, server: str, username: str, api_token: str):
        """Initialize Jira loader
        
        Args:
            server: Jira server URL
            username: Jira username/email
            api_token: Jira API token
        """
        self.client = JIRA(
            server=server,
            basic_auth=(username, api_token)
        )
    
    def load(self, source: str) -> List[Document]:
        """Load documents from a Jira issue
        
        Args:
            source: Jira issue key (e.g., 'PROJ-123')
            
        Returns:
            List of loaded documents
        """
        logger.info(f"Loading Jira issue: {source}")
        
        # Get the issue
        issue = self.client.issue(source)
        
        # Combine issue fields into a single text
        content = f"""
        Title: {issue.fields.summary}
        Description: {issue.fields.description or ''}
        Status: {issue.fields.status}
        Created: {issue.fields.created}
        Reporter: {issue.fields.reporter}
        """
        
        # Add comments
        comments = self.client.comments(issue)
        for comment in comments:
            content += f"\nComment by {comment.author} on {comment.created}:\n{comment.body}\n"
        
        # Create a document
        return [Document(
            page_content=content,
            metadata={"source": f"JIRA-{source}"}
        )]
