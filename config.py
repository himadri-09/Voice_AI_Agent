import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Azure OpenAI ──────────────────────────────────────────────────────────────
AZURE_OPENAI_API_KEY            = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT           = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_DEPLOYMENT_NAME           = os.getenv("AZURE_DEPLOYMENT_NAME")
AZURE_EMBEDDING_DEPLOYMENT_NAME = os.getenv("AZURE_EMBEDDING_DEPLOYMENT_NAME")
AZURE_API_VERSION               = os.getenv("AZURE_API_VERSION", "2024-02-15-preview")
EMBEDDING_DIMENSION             = int(os.getenv("EMBEDDING_DIMENSION", "1536"))

# ── Pinecone ──────────────────────────────────────────────────────────────────
PINECONE_API_KEY                = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME             = os.getenv("PINECONE_INDEX_NAME", "voice-rag")
PINECONE_CLOUD                  = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION                 = os.getenv("PINECONE_REGION", "us-east-1")

# ── Azure Redis ───────────────────────────────────────────────────────────────
REDIS_HOST                      = os.getenv("REDIS_HOST")           # e.g. your-cache.redis.cache.windows.net
REDIS_PORT                      = int(os.getenv("REDIS_PORT", "6380"))
REDIS_PASSWORD                  = os.getenv("REDIS_PASSWORD")
REDIS_SSL                       = os.getenv("REDIS_SSL", "true").lower() == "true"

# TTLs (seconds)
REDIS_TTL_SESSION               = int(os.getenv("REDIS_TTL_SESSION",  "1800"))  # 30 min
REDIS_TTL_SEMANTIC              = int(os.getenv("REDIS_TTL_SEMANTIC", "300"))   # 5 min
REDIS_TTL_PREFETCH              = int(os.getenv("REDIS_TTL_PREFETCH", "120"))   # 2 min

# ── Deepgram ──────────────────────────────────────────────────────────────────
DEEPGRAM_API_KEY                = os.getenv("DEEPGRAM_API_KEY")
DEEPGRAM_STT_MODEL              = "nova-2"
DEEPGRAM_TTS_MODEL              = "aura-asteria-en"

# ── Crawl ─────────────────────────────────────────────────────────────────────
MAX_PAGES                       = int(os.getenv("MAX_PAGES",        "100"))
MAX_DEPTH                       = int(os.getenv("MAX_DEPTH",        "5"))
CRAWL_CONCURRENCY               = int(os.getenv("CRAWL_CONCURRENCY","5"))

# ── Chunking ──────────────────────────────────────────────────────────────────
MIN_PAGE_WORDS                  = int(os.getenv("MIN_PAGE_WORDS",   "20"))    # skip stubs
MAX_PAGE_WORDS                  = int(os.getenv("MAX_PAGE_WORDS",   "1500"))  # warn threshold
MIN_SECTION_WORDS               = int(os.getenv("MIN_SECTION_WORDS","30"))    # skip tiny H2s
CHILD_SPLIT_THRESHOLD           = int(os.getenv("CHILD_SPLIT_THRESHOLD","300")) # pages below this = parent only

# ── Retrieval ─────────────────────────────────────────────────────────────────
TOP_K_CHILDREN                  = int(os.getenv("TOP_K_CHILDREN",  "5"))     # child chunks from Pinecone
SLOW_THINKER_N                  = int(os.getenv("SLOW_THINKER_N",  "3"))     # follow-up questions to predict

# ── Local storage ─────────────────────────────────────────────────────────────
PAGES_DIR                       = Path("pages")
PAGES_DIR.mkdir(exist_ok=True)