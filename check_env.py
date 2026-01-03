import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# Print variables
print("API Key:", os.getenv("PINECONE_API_KEY"))
print("Environment:", os.getenv("PINECONE_ENV"))
print("Index Name:", os.getenv("PINECONE_INDEX"))
print("Host:", os.getenv("PINECONE_HOST"))
