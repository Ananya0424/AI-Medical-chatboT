# store_index.py

from dotenv import load_dotenv
import os

from src.helper import (
    load_pdf_file,
    filter_to_minimal_docs,
    text_split,
    download_embeddings
)

from langchain_community.vectorstores import FAISS

# ------------------------
# Load env
# ------------------------
load_dotenv()

# ------------------------
# Load & process PDFs
# ------------------------
print("📄 Loading PDFs...")
extracted_data = load_pdf_file("data")

print("🧹 Cleaning documents...")
minimal_docs = filter_to_minimal_docs(extracted_data)

print("✂️ Splitting text into chunks...")
text_chunks = text_split(minimal_docs)

print(f"📄 Total chunks created: {len(text_chunks)}")

# ------------------------
# Embeddings
# ------------------------
print("🔢 Loading embedding model...")
embeddings = download_embeddings()

# ------------------------
# Create FAISS index
# ------------------------
print("📦 Creating FAISS index...")
vectorstore = FAISS.from_documents(
    documents=text_chunks,
    embedding=embeddings
)

# ------------------------
# Save FAISS index locally
# ------------------------
print("💾 Saving FAISS index to disk...")
vectorstore.save_local("faiss_index")

print("✅ FAISS index created and saved successfully!")
