import logging
import os

import gradio as gr
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from toml import load

from core import DocumentManager

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

# Initialize core components
document_manager = DocumentManager()
llm = ChatOpenAI(model="gpt-4o-mini")

def respond(message: str, history: list) -> str:
    """
    Generate a response based on the user query (message),
    retrieving relevant context from the document store.

    Args:
        message (str): The user's query or message.
        history (list): Chat history (unused in this code, but required by Gradio).

    Returns:
        str: The generated response.
    """
    # Search for relevant documents
    docs = document_manager.search_documents(message)
    if not docs:
        return "Please load a document or Jira issue first."

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
    submit_btn.click(document_manager.load_file, inputs=[file_input], outputs=[status_output])
    clear_btn.click(document_manager.clear_state, outputs=[file_input, status_output])
    connect_btn.click(document_manager.init_jira, inputs=[jira_server, jira_username, jira_token], outputs=[jira_status])
    load_issue_btn.click(document_manager.load_jira_issue, inputs=[issue_key], outputs=[issue_status])


if __name__ == "__main__":
    demo.launch()
