import os
import ssl

# 1. Forcefully clear the system level certificate path environment variables
if "SSL_CERT_FILE" in os.environ:
    del os.environ["SSL_CERT_FILE"]
if "SSL_CERT_DIR" in os.environ:
    del os.environ["SSL_CERT_DIR"]

# 2. Apply global SSL context bypass
ssl._create_default_https_context = ssl._create_unverified_context

from dotenv import load_dotenv
from src.helper import load_pdf_file, filter_to_minimal_docs, text_split, download_hugging_face_embeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

# Load environment variables
load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "medical-chatbot"

# Check if index exists by fetching the list of existing indexes
existing_indexes = [index.name for index in pc.list_indexes()]

if index_name not in existing_indexes:
    print(f"Creating new Pinecone index: {index_name}...")
    pc.create_index(
        name=index_name,
        dimension=384,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
    print("Index created successfully!")
else:
    print(f"Index '{index_name}' already exists. Connecting...")

# Connect to the index
index = pc.Index(index_name)

print("Starting data extraction from PDF...")
extracted_data = load_pdf_file(data='data/')

print("Filtering document metadata...")
filter_data = filter_to_minimal_docs(extracted_data)

print("Splitting text into chunks...")
text_chunks = text_split(filter_data)

print("Downloading HuggingFace embeddings...")
embeddings = download_hugging_face_embeddings()

print("Uploading vectors to Pinecone (This will take a few minutes)...")
docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    index_name=index_name,
    embedding=embeddings
)

print("All data successfully pushed to Pinecone!")