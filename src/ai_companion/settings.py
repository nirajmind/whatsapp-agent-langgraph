from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore", env_file_encoding="utf-8")

    GROQ_API_KEY: str = ""
    ELEVENLABS_API_KEY: str = ""
    ELEVENLABS_VOICE_ID: str = ""
    TOGETHER_API_KEY: str = ""
    OPENAI_API_KEY: str = ""
    OPENAI_BASE_URL: str = ""

#    QDRANT_API_KEY: str
#    QDRANT_URL: str
#    QDRANT_PORT: str = "6333"
#    QDRANT_HOST: str | None = None
    MONGO_URI: str = ""
    DATABASE_NAME: str = ""
    COLLECTION_NAME: str = ""
    MONGODB_CHECKPOINT_DB_NAME: str = ""
    MONGODB_CHECKPOINT_COLLECTION_NAME: str = ""

    TEXT_MODEL_NAME: str = ""
    SMALL_TEXT_MODEL_NAME: str = "grok-3-mini-fast-latest"
    STT_MODEL_NAME: str = "whisper-1"
    TTS_MODEL_NAME: str = "eleven_flash_v2_5"
    TTI_MODEL_NAME: str = "black-forest-labs/FLUX.1-schnell-Free"
    ITT_MODEL_NAME: str = "gpt-4o"

    MEMORY_TOP_K: int = 3
    ROUTER_MESSAGES_TO_ANALYZE: int = 3
    TOTAL_MESSAGES_SUMMARY_TRIGGER: int = 20
    TOTAL_MESSAGES_AFTER_SUMMARY: int = 5

    SHORT_TERM_MEMORY_DB_PATH: str = "/app/data/memory.db"
    EMBEDDER_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"


settings = Settings()
