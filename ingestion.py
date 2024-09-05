import os
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.vectorstores.base import VectorStoreRetriever
from langchain_openai import OpenAIEmbeddings

load_dotenv()

# Define persistence directory
persist_directory = "./.chroma"

# URLs to be processed
urls = [
    "https://lilianweng.github.io/posts/2023-06-23-agent/",
    "https://lilianweng.github.io/posts/2023-03-15-prompt-engineering/",
    "https://lilianweng.github.io/posts/2023-10-25-adv-attack-llm/",
]

# Load the documents from the URLs
docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]

# Split the documents into chunks
text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=250, chunk_overlap=0
)
doc_splits = text_splitter.split_documents(docs_list)

# Check if the Chroma vector store already exists
if not os.path.exists(persist_directory):
    print("Chroma vector store not found. Creating a new one...")

    # Initialize Chroma and insert the documents if the store doesn't exist
    vectorstore = Chroma.from_documents(
        documents=doc_splits,
        collection_name="rag-chroma",
        embedding=OpenAIEmbeddings(),
        persist_directory=persist_directory,
    )
else:
    print("Chroma vector store already exists. Skipping document insertion.")

# Load the retriever from the Chroma store (existing or newly created)
retriever: VectorStoreRetriever = Chroma(
    collection_name="rag-chroma",
    persist_directory=persist_directory,
    embedding_function=OpenAIEmbeddings(),
).as_retriever()
