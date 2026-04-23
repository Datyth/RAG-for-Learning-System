from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List, Optional

class RecursiveChunker:
    def __init__(
        self, 
        chunk_size: int = 500, 
        chunk_overlap: int = 50, 
        separators: Optional[List[str]] = None
    ):

        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators,
            is_separator_regex=False
        )

    def split_documents(self, documents: List[Document]) -> List[Document]:
        print(f" Starting Recursive Chunking: size={self.chunk_size}, overlap={self.chunk_overlap}")
        if not documents:
            print("Error: No documents provided for chunking.")
            return []
        chunks = self.text_splitter.split_documents(documents)
        print(f" {len(documents)} documents split into {len(chunks)} chunks.")
        return chunks

    def split_text(self, text: str) -> List[str]:
        return self.text_splitter.split_text(text)


class SemanticChunker(RecursiveChunker):
    def __init__(
        self, 
        chunk_size: int = 500, 
        chunk_overlap: int = 50, 
        separators: Optional[List[str]] = None,
        embedding = None
    ):
        super().__init__(chunk_size, chunk_overlap, separators)