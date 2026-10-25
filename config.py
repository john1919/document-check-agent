"""All model names, runtime limits, and retrieval settings live here."""
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).with_name('.env'))
CHAT_MODEL = 'gpt-4.1-mini'
EMBEDDING_MODEL = 'text-embedding-3-small'
TEMPERATURE = 0
CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200
TOP_K = 5
MAX_RETRIES = 0  # Fail fast on API errors; separate from the evidence-search retry below.
MAX_SEARCH_ATTEMPTS = 2  # Original search plus exactly one reworded search.
MAX_RULES = 50
MAX_GRAPH_STEPS = 270  # Covers 50 rules even when every rule needs a second search.
TIMEOUT_RETRIES = 1  # Retry timeouts only; other API errors surface immediately.
REQUEST_TIMEOUT = 60  # Seconds per OpenAI request, for chat and embeddings.
RRF_CONSTANT = 60
MAX_UPLOAD_MB = 10
