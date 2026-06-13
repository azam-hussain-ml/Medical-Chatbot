import os
import ssl

# Global SSL bypass for HuggingFace downloads within this module
ssl._create_default_https_context = ssl._create_unverified_context

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from typing import List
from langchain_core.documents import Document

# Extract data from the PDF files within the specified directory
def load_pdf_file(data):
    loader = DirectoryLoader(
        data,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    documents = loader.load()
    return documents


# Filter and minimize document structure by retaining only essential metadata
def filter_to_minimal_docs(extracted_data):
    docs = extracted_data
    minimal_docs: List[Document] = []
    
    for doc in docs:
        src = doc.metadata.get('source', 'unknown')
        
        minimal_docs.append(
            Document(
                page_content=doc.page_content,
                metadata={"source": src}
            )
        )
    return minimal_docs


# Split the documents into smaller text chunks for better embedding generation
def text_split(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20
    )
    text_chunks = text_splitter.split_documents(extracted_data)
    return text_chunks


# Download and initialize the pre-trained HuggingFace embeddings model
def download_hugging_face_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embeddings