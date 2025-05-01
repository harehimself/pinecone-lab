import os

# Get API key directly from environment variable
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY environment variable not set")

# Default settings
DEFAULT_CLOUD = "aws"
DEFAULT_REGION = "us-east-1"
DEFAULT_DIMENSION = 1536
DEFAULT_METRIC = "cosine"
DEFAULT_NAMESPACE = "default"

# Sample models for embedding
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
SENTENCE_TRANSFORMER_MODEL = "all-MiniLM-L6-v2"