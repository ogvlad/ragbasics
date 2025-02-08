# Document Q&A with RAG

A Retrieval-Augmented Generation (RAG) application that allows you to ask questions about your documents. Supports PDF files, Word documents, and Jira issues.

## Features

- Upload and process PDF and Word documents
- Connect to Jira and load issue content
- Ask questions about your documents using natural language
- Modern web interface built with Gradio
- Uses OpenAI's models for embeddings and chat completion

## Prerequisites

- Python 3.8 or higher
- OpenAI API key
- Jira API token (optional, only for Jira integration)

## Installation

### Windows

1. Create a virtual environment:
```powershell
# Navigate to project directory
cd path\to\project

# Create virtual environment
python -m venv venv

# Activate virtual environment
.\venv\Scripts\activate
```

### Linux/macOS

1. Create a virtual environment:
```bash
# Navigate to project directory
cd path/to/project

# Create virtual environment
python -m venv venv

# Activate virtual environment
source venv/bin/activate
```

### Install Dependencies

With the virtual environment activated:
```bash
pip install -r requirements.txt
```

## Configuration

1. Set up your OpenAI API key:
```bash
# Windows (PowerShell)
$env:OPENAI_API_KEY="your-api-key-here"

# Linux/macOS
export OPENAI_API_KEY="your-api-key-here"
```

2. (Optional) For Jira integration, set up your API token:
```bash
# Windows (PowerShell)
$env:JIRA_API_TOKEN="your-jira-api-token-here"

# Linux/macOS
export JIRA_API_TOKEN="your-jira-api-token-here"
```

## Running the Application

1. Make sure your virtual environment is activated
2. Run the application:
```bash
python app.py
```
3. Open your web browser and navigate to the URL shown in the terminal (typically http://127.0.0.1:7860)

## Usage

### Document Upload
1. Go to the "Document Upload" tab
2. Click "Upload PDF or Word Document" and select your file
3. Click "Submit" to process the document
4. Ask questions about your document in the chat interface

### Jira Integration
1. Go to the "Jira Connection" tab
2. Enter your Jira credentials:
   - Server URL (e.g., https://your-domain.atlassian.net)
   - Username/Email
   - API Token
3. Click "Connect to Jira"
4. Enter a Jira issue key (e.g., PROJ-123) and click "Load Issue"
5. Ask questions about the loaded issue in the chat interface

## Project Structure

```
.
├── app.py              # Main application file
├── requirements.txt    # Python dependencies
├── _db/               # Persistent vector store directory
├── core/              # Core functionality
│   ├── __init__.py
│   ├── document_store.py
│   └── manager.py
└── loaders/           # Document loaders
    ├── __init__.py
    ├── base_loader.py
    ├── pdf_loader.py
    ├── word_loader.py
    └── jira_loader.py
```

## Data Persistence

The application uses ChromaDB for storing document embeddings. The vector store is persistent and stored in the `_db` directory. This means:
- Your uploaded documents and their embeddings are preserved between application restarts
- You don't need to re-upload documents every time you start the application
- The `Clear` button in the UI will remove all documents from the store

## Troubleshooting

1. **ModuleNotFoundError**: Make sure you have activated the virtual environment and installed all dependencies
2. **OpenAI API Error**: Check that your OpenAI API key is set correctly
3. **Jira Connection Error**: Verify your Jira credentials and API token
4. **File Loading Error**: Ensure you're uploading supported file types (PDF or Word documents)
