# app_folder/config.py
import os
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from fastembed import TextEmbedding
from groq import Groq

load_dotenv()

# Load environment variables
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
SERPER_API_KEY = os.getenv("SERPER_API_KEY")

# Ensure all keys are present
if not all([QDRANT_URL, QDRANT_API_KEY, GROQ_API_KEY]):
    raise ValueError("Missing required API keys in .env file")

# Setup
COLLECTION_NAME = "bhagavad-gita"
CLIENT = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, prefer_grpc=True)
EMBED_MODEL = TextEmbedding(model_name="thenlper/gte-large")
GROQ_CLIENT = Groq(api_key=GROQ_API_KEY)
GITA_MODEL = "deepseek-r1-distill-llama-70b"
ARTICLE_MODEL = "gemma2-9b-it"

# Cache system
RESOURCE_CACHE = {}
GITA_CACHE = {}