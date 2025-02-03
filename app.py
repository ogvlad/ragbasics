import os

import gradio as gr

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter

import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

llm = ChatOpenAI(model="gpt-4o-mini")
embed_model = OpenAIEmbeddings()


# File processing function
def load_files(file_path: str):
    global vector_store

    logger.info(f"Loading file: {file_path}")

    # Load the document
    document_loader = PyPDFLoader(file_path)
    documents = document_loader.load()
    logger.info(f"Loaded {len(documents)} documents")

    # Split the document into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    logger.info(f"Split into {len(chunks)} chunks")

    logger.info("Creating vector store")
    # Create the vector store
    vector_store = Chroma.from_documents(documents=chunks, embedding=OpenAIEmbeddings())
    logger.info("Vector store created")


# Respond function
def respond(message, history):
    logger.info("Retrieving from vector store")
    retriever = vector_store.as_retriever()
    logger.info("Retrieved from vector store")

    prompt = """
            Answer the question in the below context:
            {context}

            Question: {question}
        """

    prompt_template = ChatPromptTemplate.from_template(prompt)

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt_template
        | llm
        | StrOutputParser()
    )

    response = chain.invoke( message)
    return response


# Clear function
def clear_state():
    global vector_index
    vector_index = None
    return [None, None, None]


# UI Setup
with gr.Blocks(
    theme=gr.themes.Default(
        primary_hue="green",
        secondary_hue="blue",
        # font=[gr.themes.GoogleFont("Poppins")],
    ),
    css="footer {visibility: hidden}",
) as demo:
    gr.Markdown("RAG Starter App")
    with gr.Row():
        with gr.Column(scale=1):
            file_input = gr.File(
                file_count="single", type="filepath", label="Upload PDF Document"
            )
            with gr.Row():
                btn = gr.Button("Submit", variant="primary")
                clear = gr.Button("Clear")
            output = gr.Textbox(label="Status")
        with gr.Column(scale=3):
            chatbot = gr.ChatInterface(
                fn=respond,
                chatbot=gr.Chatbot(height=300),
                theme="soft",
                show_progress="full",
                textbox=gr.Textbox(
                    placeholder="Ask questions about the uploaded document!",
                    container=False,
                ),
            )

    # Set up Gradio interactions
    btn.click(fn=load_files, inputs=file_input, outputs=output)
    clear.click(
        fn=clear_state,  # Use the clear_state function
        outputs=[file_input, output],
    )

# Launch the demo
if __name__ == "__main__":
    demo.launch()
