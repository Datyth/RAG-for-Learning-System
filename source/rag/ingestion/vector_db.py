import os
import uuid  # Thư viện để sinh ID tự động
from pathlib import Path
from typing import Union, Optional, Type, List

from langchain_chroma import Chroma
from langchain_community.vectorstores import FAISS
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_huggingface import HuggingFaceEmbeddings

from file_loader import Loader

project_root = Path(__file__).resolve().parent.parent.parent.parent

class VectorDB:
    def __init__(self, 
                 documents = None, 
                 vector_db: Type[Union[Chroma, FAISS, QdrantVectorStore]] = QdrantVectorStore,
                 embedding = None,):
        
        self.vector_db = vector_db
        self.embedding = embedding or HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        if self.vector_db == QdrantVectorStore:
            self.persist_directory = project_root / "vector_db" / "qdrant_data"
        elif self.vector_db == Chroma:
            self.persist_directory = project_root / "vector_db" / "chroma_data"
        elif self.vector_db == FAISS:
            self.persist_directory = project_root / "vector_db" / "faiss_data"
        self.db = self._load_or_build_db(documents)

    def _load_or_build_db(self, documents):
        if self.persist_directory and os.path.exists(self.persist_directory):
            print(f"Loading available VectorDB from: {self.persist_directory}")

            if self.vector_db == QdrantVectorStore:
                client = QdrantClient(path=self.persist_directory)
                return QdrantVectorStore(
                    client=client,
                    collection_name="my_collection",
                    embedding=self.embedding
                )
            
            elif self.vector_db == Chroma:
                return Chroma(
                    persist_directory=self.persist_directory,
                    embedding_function=self.embedding
                )
                
            elif self.vector_db == FAISS:
                return FAISS.load_local(
                    self.persist_directory,
                    self.embedding,
                    index_name="index",
                    allow_dangerous_deserialization=True 
                )

        if documents is None:
            print("Warning: Cannot build DB. No persistent directory found and no documents provided.")
            return None
        
        print(f"Building new VectorDB in {self.persist_directory}")


        initial_ids = [str(uuid.uuid4()) for _ in range(len(documents))]

        if self.vector_db == QdrantVectorStore:
            db = QdrantVectorStore.from_documents(
                documents=documents,
                embedding=self.embedding,
                path=self.persist_directory,
                collection_name="my_collection",
                ids=initial_ids  
            )
            
        elif self.vector_db == Chroma:
            db = Chroma.from_documents(
                documents=documents,
                embedding=self.embedding,
                persist_directory=self.persist_directory,
                ids=initial_ids 
            )
        
        elif self.vector_db == FAISS:
            db = FAISS.from_documents(
                documents=documents, 
                embedding=self.embedding,
                ids=initial_ids  
            )
            if self.persist_directory:
                print(f"Saving vectors into: {self.persist_directory}")
                os.makedirs(self.persist_directory, exist_ok=True)
                db.save_local(self.persist_directory, index_name="index")
        
        return db
    
    def add_item(self, document) -> str:
        if self.db is None:
            raise ValueError("Vector database is not initialized.")
        
        new_id = str(uuid.uuid4())
        self.db.add_documents([document], ids=[new_id])
        return new_id
    
    def delete_item(self, item_id: str):
        if self.db is None:
            raise ValueError("Vector database is not initialized.")
        
        self.db.delete(ids=[item_id])
    
    def get_retriever(self, search_type: str= "similarity", search_kwargs: dict = None):
        if search_kwargs is None:
            search_kwargs = {"k": 10}
            
        if self.db is None:
            raise ValueError("Vector database is not initialized.")
        return self.db.as_retriever(search_type=search_type, search_kwargs=search_kwargs)
    
if __name__ == "__main__":
    dir_path = "data"
    split_kwargs: dict = {
                    "chunk_size": 500,
                    "chunk_overlap": 50
                }
    loader = Loader(file_type = 'pdf', split_kwargs = split_kwargs)
    docs = loader.load_dir(dir_path = dir_path)
    print(len(docs))
    for i in range(6):
        print(docs[i].page_content[:])


    vector_db = VectorDB(documents=docs, vector_db=QdrantVectorStore)

    if vector_db.db is not None and hasattr(vector_db.db, 'client'):
            vector_db.db.client.close()