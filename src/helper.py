from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from typing import List
from langchain_core.documents import Document


# ---------------------------
# Load PDFs
# ---------------------------
def load_pdf_file(data_path: str) -> List[Document]:
    loader = DirectoryLoader(
        path=data_path,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    documents = loader.load()

    if not documents:
        raise ValueError("❌ No PDF files found in the data folder!")

    return documents


# ---------------------------
# Keep only minimal metadata
# ---------------------------
def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    minimal_docs = []

    for doc in docs:
        minimal_docs.append(
            Document(
                page_content=doc.page_content,
                metadata={
                    "source": doc.metadata.get("source", "unknown")
                }
            )
        )

    return minimal_docs


# ---------------------------
# Split text into chunks
# ---------------------------
def text_split(docs: List[Document]) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    return splitter.split_documents(docs)


# ---------------------------
# HuggingFace embeddings (FREE)
# ---------------------------
def download_embeddings():
    model_name = "sentence-transformers/all-MiniLM-L6-v2"

    embeddings = HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={"device": "cpu"}  # avoids CUDA errors
    )

    return embeddings