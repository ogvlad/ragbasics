from toml import load
from typing import List
from chunking.recursive_character_text_splitter import RecursiveCharacterTextSplitterStrategy
from langchain.docstore.document import Document

# load pypromect.toml
with open("pyproject.toml", "r") as f:
    config = load(f)


class Chunker:
    def __init__(self):
        if config["chunking"]["strategy"] == "recursive_character_text_splitter":
            self.chunking_strategy = RecursiveCharacterTextSplitterStrategy(
                chunk_size=config["chunking"]["chunk_size"],
                chunk_overlap=config["chunking"]["chunk_overlap"],
            )
        else:
            raise ValueError(f"Invalid chunking strategy: {config['chunking']['strategy']}")

    def chunk(self, documents: List[Document]) -> List[Document]:
        return self.chunking_strategy.chunk(documents)
