import logging
import os

import gradio as gr
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from toml import load

from loaders import PDFDocumentLoader, WordDocumentLoader, JiraLoader

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Default Jira settings
DEFAULT_JIRA_URL = "https://eassessment.atlassian.net"
DEFAULT_JIRA_USER = "vlad.ogay@cirrusassessment.com"
JIRA_API_TOKEN = os.getenv("JIRA_API_TOKEN", "")

# load pyproject.toml
with open("pyproject.toml", "r") as f:
    config = load(f)

# Global objects
vector_store = None
jira_loader = None

# Initialize language model and embeddings
llm = ChatOpenAI(model="gpt-4o-mini")
embedding = OpenAIEmbeddings()

def load_files(file_path: str) -> str:
    """
    Load a PDF or Word document from the given file path, split it into chunks,
    and store these chunks in a global vector store (Chroma).

    Args:
        file_path (str): The path to the PDF or Word document.

    Returns:
        str: A status message indicating the file was loaded successfully.
    """
    global vector_store

    logger.info(f"Loading file: {file_path}")

    # Determine file type and use appropriate loader
    file_extension = os.path.splitext(file_path)[1].lower()
    
    if file_extension == '.pdf':
        loader = PDFDocumentLoader()
    elif file_extension in ['.docx', '.doc']:
        loader = WordDocumentLoader()
    else:
        return f"Unsupported file type: {file_extension}. Please upload a PDF or Word document."

    try:
        # Load the document
        documents = loader.load(file_path)
        logger.info(f"Loaded {len(documents)} document(s)")

        # Split the document into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)

        # Create or update the vector store
        if vector_store is None:
            vector_store = Chroma.from_documents(
                documents=chunks,
                embedding=embedding
            )
        else:
            vector_store.add_documents(chunks)

        return "File loaded successfully. You can now ask questions about the document."
    except Exception as e:
        logger.error(f"Error loading file: {str(e)}")
        return f"Error loading file: {str(e)}"


def init_jira(server: str, username: str, api_token: str) -> str:
    """
    Initialize Jira client with the provided credentials.

    Args:
        server (str): Jira server URL
        username (str): Jira username (email)
        api_token (str): Jira API token

    Returns:
        str: Status message
    """
    global jira_loader
    try:
        jira_loader = JiraLoader(server, username, api_token)
        return "Successfully connected to Jira"
    except Exception as e:
        logger.error(f"Failed to connect to Jira: {str(e)}")
        return f"Failed to connect to Jira: {str(e)}"


def load_jira_issue(issue_key: str) -> str:
    """
    Load a Jira issue and its comments, convert them to a document format.

    Args:
        issue_key (str): Jira issue key (e.g., 'PROJ-123')

    Returns:
        str: Status message
    """
    global vector_store, jira_loader

    if not jira_loader:
        return "Please connect to Jira first"

    try:
        # Load the issue using JiraLoader
        documents = jira_loader.load(issue_key)
        
        # Split the document into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)

        # Create or update the vector store
        if vector_store is None:
            vector_store = Chroma.from_documents(
                documents=chunks,
                embedding=embedding
            )
        else:
            vector_store.add_documents(chunks)

        return f"Successfully loaded Jira issue {issue_key}"
    except Exception as e:
        logger.error(f"Error loading Jira issue: {str(e)}")
        return f"Error loading Jira issue: {str(e)}"


def respond(message: str, history: list) -> str:
    """
    Generate a response based on the user query (message),
    retrieving relevant context from the global vector store.

    Args:
        message (str): The user's query or message.
        history (list): Chat history (unused in this code, but required by Gradio).

    Returns:
        str: The generated response.
    """
    if vector_store is None:
        return "Please load a document or Jira issue first."

    # Search for relevant documents
    docs = vector_store.similarity_search(message)

    # Create a prompt template
    template = """Answer the question based on the following context:

    Context:
    {context}

    Question:
    {question}

    Answer:"""

    prompt = ChatPromptTemplate.from_template(template)

    # Create the chain
    chain = (
        {"context": lambda x: x, "question": lambda x: x}
        | prompt
        | llm
        | StrOutputParser()
    )

    # Run the chain
    return chain.invoke(
        {
            "context": "\n\n".join([doc.page_content for doc in docs]),
            "question": message,
        }
    )


def clear_state() -> list:
    """
    Clear the global vector store reference and reset the UI fields.

    Returns:
        list: A list of values for resetting Gradio components.
    """
    global vector_store
    vector_store = None
    return [None, "State cleared. Ready for new documents."]  # Reset file input and status textbox


# Gradio UI Setup
with gr.Blocks(
    theme=gr.themes.Default(
        primary_hue="blue",
        secondary_hue="neutral",
    ),
    title="Document Q&A",
) as demo:
    gr.Markdown("# Document Q&A")
    
    with gr.Tab("Document Upload"):
        with gr.Row():
            with gr.Column(scale=1):
                file_input = gr.File(
                    file_count="single", type="filepath", label="Upload PDF or Word Document"
                )
                with gr.Row():
                    submit_btn = gr.Button("Submit", variant="primary")
                    clear_btn = gr.Button("Clear", variant="secondary")
                status_output = gr.Textbox(label="Status", interactive=False)
            with gr.Column(scale=3):
                chatbot = gr.ChatInterface(
                    fn=respond,
                    chatbot=gr.Chatbot(height=800),
                    theme="soft",
                    show_progress="full",
                    textbox=gr.Textbox(
                        placeholder="Ask a question about the document...",
                        container=False,
                        scale=7,
                    ),
                )
    
    with gr.Tab("Jira Connection"):
        with gr.Row():
            with gr.Column():
                jira_server = gr.Textbox(label="Jira Server URL", 
                                       placeholder="https://your-domain.atlassian.net",
                                       value=DEFAULT_JIRA_URL)
                jira_username = gr.Textbox(label="Username/Email",
                                         value=DEFAULT_JIRA_USER)
                jira_token = gr.Textbox(label="API Token", 
                                      type="password",
                                      value=JIRA_API_TOKEN,
                                      placeholder="Enter token or set JIRA_API_TOKEN environment variable")
                connect_btn = gr.Button("Connect to Jira")
                jira_status = gr.Textbox(label="Connection Status", interactive=False)
        
        with gr.Row():
            with gr.Column():
                issue_key = gr.Textbox(label="Jira Issue Key", placeholder="e.g., PROJ-123")
                load_issue_btn = gr.Button("Load Issue")
                issue_status = gr.Textbox(label="Loading Status", interactive=False)

    # Event handlers
    submit_btn.click(load_files, inputs=[file_input], outputs=[status_output])
    clear_btn.click(clear_state, outputs=[file_input, status_output])
    connect_btn.click(init_jira, inputs=[jira_server, jira_username, jira_token], outputs=[jira_status])
    load_issue_btn.click(load_jira_issue, inputs=[issue_key], outputs=[issue_status])


if __name__ == "__main__":
    demo.launch()
