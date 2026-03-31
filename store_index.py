# store_index.py
from src.helper import load_pdf_file, filter_to_minimal_docs, text_split, download_embeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
import os

# Load variables
load_dotenv()

# Step 1: Load and process PDFs
print("📄 Loading PDFs from 'data' folder...")
extracted_data = load_pdf_file("data")

print("🧹 Cleaning documents and metadata...")
minimal_docs = filter_to_minimal_docs(extracted_data)

print("✂️ Splitting text into chunks...")
text_chunks = text_split(minimal_docs)
print(f"✅ Total chunks created: {len(text_chunks)}")

# Step 2: Download embeddings
print("🔢 Loading embedding model...")
embeddings = download_embeddings()

# Step 3: Create and Save FAISS index
print("📦 Creating FAISS index... Please wait.")
vectorstore = FAISS.from_documents(
    documents=text_chunks,
    embedding=embeddings
)

print("💾 Saving FAISS index to 'faiss_index'...")
vectorstore.save_local("faiss_index")

print("🎉 FAISS index created and saved successfully!")
