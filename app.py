import logging
import os

import gradio as gr
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from chunking import Chunker

from toml import load

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
        secondary_hue="gray",
    ),
) as demo:
    gr.Markdown("# RAG Starter App")
    with gr.Row():
        with gr.Column(scale=1):
            file_input = gr.File(
                file_count="single", type="filepath", label="Upload PDF or Word Document"
            )
            with gr.Row():
                submit_btn = gr.Button("Submit", variant="primary")
                clear_btn = gr.Button("Clear")

            status_output = gr.Textbox(label="Status")

        with gr.Column(scale=3):
            chatbot = gr.ChatInterface(
                fn=respond,
                chatbot=gr.Chatbot(height=800),
                theme="soft",
                show_progress="full",
                textbox=gr.Textbox(
                    placeholder="Ask questions about the uploaded document!",
                    container=False,
                ),
            )

    # Set up Gradio interactions
    submit_btn.click(fn=load_files, inputs=file_input, outputs=status_output)
    clear_btn.click(
        fn=clear_state,
        outputs=[file_input, status_output],
    )

if __name__ == "__main__":
    demo.launch()
