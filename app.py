import logging
import os

import gradio as gr
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from chunking import Chunker
from jira import JIRA
from toml import load
from langchain import LLMChain  # Corrected import

# load pyproject.toml
with open("pyproject.toml", "r") as f:
    config = load(f)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Global objects (kept for functionality)
vector_store = None
jira_client = None

# Initialize language model and embeddings
llm = ChatOpenAI(model="gpt-4o-mini")
embedding = OpenAIEmbeddings()

# Default Jira settings
DEFAULT_JIRA_URL = "https://eassessment.atlassian.net"
DEFAULT_JIRA_USER = "vlad.ogay@cirrusassessment.com"
JIRA_API_TOKEN = os.getenv("JIRA_API_TOKEN", "")

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
        loader = PyPDFLoader(file_path)
    elif file_extension in ['.docx', '.doc']:
        loader = Docx2txtLoader(file_path)
    else:
        return f"Unsupported file type: {file_extension}. Please upload a PDF or Word document."

    # Load the document
    documents = loader.load()
    logger.info(f"Loaded {len(documents)} document(s)")

    # Split the document into chunks
    chunks = Chunker().chunk(documents)
    logger.info(f"Split into {len(chunks)} chunk(s)")

    logger.info("Creating vector store...")
    vector_store = Chroma.from_documents(documents=chunks, embedding=embedding)
    logger.info("Vector store created successfully.")

    return "File loaded successfully. You can now ask questions about the document."


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
    global jira_client
    try:
        jira_client = JIRA(
            server=server,
            basic_auth=(username, api_token)
        )
        return "Successfully connected to Jira"
    except Exception as e:
        return f"Failed to connect to Jira: {str(e)}"


def load_jira_issue(issue_key: str) -> str:
    """
    Load a Jira issue and its comments, convert them to a document format.

    Args:
        issue_key (str): Jira issue key (e.g., 'PROJ-123')

    Returns:
        str: Status message
    """
    global vector_store, jira_client

    if not jira_client:
        return "Please connect to Jira first"

    try:
        # Get the issue
        issue = jira_client.issue(issue_key)
        
        # Combine issue fields into a single text
        content = f"""
        Title: {issue.fields.summary}
        Description: {issue.fields.description or ''}
        Status: {issue.fields.status}
        Created: {issue.fields.created}
        Reporter: {issue.fields.reporter}
        """

        # Add comments
        comments = jira_client.comments(issue)
        for comment in comments:
            content += f"\nComment by {comment.author} on {comment.created}:\n{comment.body}\n"

        # Create a document
        documents = [Document(page_content=content, metadata={"source": f"JIRA-{issue_key}"})]
        
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
        logger.warning("Vector store is not initialized. Please upload a PDF first.")
        return "No document loaded. Please upload a PDF first."

    logger.info("Retrieving from vector store...")
    retriever = vector_store.as_retriever()
    logger.info("Retrieved from vector store.")

    prompt_template = ChatPromptTemplate.from_template(config["rag_prompt"]["prompt_template"])

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt_template
        | llm
        | StrOutputParser()
    )

    response = chain.invoke(message)
    return response


def clear_state():
    """
    Clear the global vector store reference and reset the UI fields.

    Returns:
        list: A list of values for resetting Gradio components.
    """
    global vector_store
    vector_store = None
    return [None, None]  # Reset file input and status textbox


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
    submit_btn.click(load_files, inputs=[file_input], outputs=[chatbot])
    clear_btn.click(clear_state, outputs=[file_input, chatbot])
    connect_btn.click(init_jira, inputs=[jira_server, jira_username, jira_token], outputs=[jira_status])
    load_issue_btn.click(load_jira_issue, inputs=[issue_key], outputs=[issue_status])

if __name__ == "__main__":
    demo.launch()
