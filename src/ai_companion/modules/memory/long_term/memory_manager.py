from functools import lru_cache
import logging
import uuid
import httpx
from datetime import datetime
from typing import List, Optional
import re  # Import re if not already present
import chainlit as cl
import os

from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field

from ai_companion.core.prompts import MEMORY_ANALYSIS_PROMPT
from ai_companion.modules.memory.long_term.vector_store import get_vector_store
from ai_companion.settings import settings


class MemoryAnalysis(BaseModel):
    """Result of analyzing a message for memory-worthy content."""
    is_important: bool = Field(..., description="Whether the message is important enough to be stored as a memory")
    formatted_memory: Optional[str] = Field(None, description="The formatted memory to be stored")
    user_name: Optional[str] = Field(None, description="The extracted user's name, if present")


class MemoryManager:
    """Manager class for handling long-term memory operations."""

    REQUIRED_ENV_VARS = ["GROQ_API_KEY", "ELEVENLABS_API_KEY", "TOGETHER_API_KEY"] # Example from your settings
    
    _instance: Optional["MemoryManager"] = None
    _is_actually_initialized: bool = False
    
    def __new__(cls) -> "MemoryManager":
        if cls._instance is None:
            cl.logger.info("MEMORY_MANAGER - Creating new MemoryManager instance (singleton).")
            cls._instance = super().__new__(cls)
            cls._instance._one_time_init() # Call dedicated init method for first time
        else:
            cl.logger.info("MEMORY_MANAGER - Reusing existing MemoryManager instance (singleton).")
        return cls._instance

    def _one_time_init(self) -> None:
        """Initializes components that only need to be set up once."""
        self._validate_env_vars()
        cl.logger.info("MEMORY_MANAGER - Environment variables validated for actual init.")

        self.vector_store = get_vector_store() # This will get the VectorStore singleton
        cl.logger.info(f"MEMORY_MANAGER - get_vector_store returned: {self.vector_store}")
        if self.vector_store is None:
            cl.logger.error("MEMORY_MANAGER - vector_store is None after get_vector_store()! This is critical.")
            raise ValueError("Vector store failed to initialize within MemoryManager.")

        self.logger = logging.getLogger(__name__) # Use this for logging within methods
        cl.logger.info(f"MEMORY_MANAGER - Logger initialized with grok model - {settings.TEXT_MODEL_NAME}.")
        
        self.model = settings.TEXT_MODEL_NAME  # Use the text model from settings
        self.api_key = settings.GROQ_API_KEY
        cl.logger.info(f"MEMORY_MANAGER - Using model: {self.model} with API key: {self.api_key}")
        self.base_url = "https://api.x.ai/v1/chat/completions"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "Ava-AI-Companion/1.0"
        }
        self._is_actually_initialized = True
        cl.logger.info("MEMORY_MANAGER - MemoryManager initialized successfully.")

    def __init__(self) -> None:
        # This __init__ will be called on every instantiation, even if __new__ returns
        # an existing instance. Keep it minimal.
        pass # The real initialization happens in _one_time_init

    def _validate_env_vars(self) -> None:
        """Validate that all required environment variables are set."""
        missing_vars = [var for var in self.REQUIRED_ENV_VARS if not os.getenv(var)]
        if missing_vars:
            cl.logger.error(f"MEMORY_MANAGER - Missing required environment variables: {', '.join(missing_vars)}")
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

    async def _analyze_memory(self, message: str) -> MemoryAnalysis:
        """Analyze a message to determine importance and format if needed."""
        prompt = MEMORY_ANALYSIS_PROMPT.format(message=message)
        
        cl.logger.info(f"Making request to: {self.base_url}")
        headers_to_log = self.headers.copy()
        if "Authorization" in headers_to_log:
            headers_to_log["Authorization"] = "***REDACTED***"
        cl.logger.info(f"Headers: {headers_to_log}")
        request_body = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1
        }
        body_to_log = request_body.copy()
        if "messages" in body_to_log and body_to_log["messages"] and isinstance(body_to_log["messages"][0], dict) and "content" in body_to_log["messages"][0]:
            # Log the prompt content but not the full body
            cl.logger.info(f"Request prompt: {body_to_log['messages'][0]['content']}")
        else:
            cl.logger.info("Request body (without sensitive info): <redacted>")

        async with httpx.AsyncClient(
            timeout=30.0,
            follow_redirects=True,
            verify=True
        ) as client:
            try:
                response = await client.post(
                    self.base_url,
                    headers=self.headers,
                    json={
                        "model": self.model,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.1
                    }
                )
                cl.logger.info(f"Response status: {response.status_code}")
                cl.logger.info(f"Response headers: {response.headers}")
                cl.logger.info(f"Response body: {response.text}")

                response.raise_for_status()
                result = response.json()
                formatted_memory = result["choices"][0]["message"]["content"]
                user_name = None
                name_match = re.search(r"i'm\s+(\w+)|my\s+name\s+is\s+(\w+)", message, re.IGNORECASE)
                if name_match:
                    user_name = name_match.group(1) or name_match.group(2)

                return MemoryAnalysis(
                    is_important=True,  # You'll need to parse this from the response
                    formatted_memory=result["choices"][0]["message"]["content"]
                )
            except httpx.HTTPError as e:
                cl.logger.error(f"HTTP error occurred: {str(e)}")
                cl.logger.error(f"Request URL: {e.request.url}")
                cl.logger.error(f"Request headers: {e.request.headers}")
                cl.logger.error(f"Request body: {e.request.content}")
                raise

    async def extract_and_store_memories(self, message: BaseMessage) -> None:
        """Extract important information from a message and store in vector store."""
        if message.type != "human":
            return

        # Analyze the message for importance and formatting
        analysis = await self._analyze_memory(message.content)
        if analysis.is_important and analysis.formatted_memory:
            # Check if similar memory exists
            similar = self.vector_store.find_similar_memory(analysis.formatted_memory)
            if similar:
                # Skip storage if we already have a similar memory
                cl.logger.info(f"Similar memory already exists: '{analysis.formatted_memory}'")
                return

            # Store new memory with user_name in metadata
            metadata = {
                "id": str(uuid.uuid4()),
                "timestamp": datetime.now().isoformat(),
            }
            if analysis.user_name:
                metadata["user_name"] = analysis.user_name

            cl.logger.info(f"Storing new memory: '{analysis.formatted_memory}' with metadata: {metadata}")
            self.vector_store.store_memory(
                text=analysis.formatted_memory,
                metadata=metadata,
            )

    def get_relevant_memories(self, context: str) -> List[str]:
        """Retrieve relevant memories based on the current context."""
        memories = self.vector_store.search_memories(context, k=settings.MEMORY_TOP_K)
        if memories:
            for memory in memories:
                self.logger.debug(f"Memory: '{memory.text}' (score: {memory.score:.2f})")
        return [memory.text for memory in memories]

    def format_memories_for_prompt(self, memories: List[str]) -> str:
        """Format retrieved memories as bullet points."""
        if not memories:
            return ""
        return "\n".join(f"- {memory}" for memory in memories)

@lru_cache(maxsize=1)
def get_memory_manager() -> MemoryManager:
    """Get a MemoryManager instance."""
    cl.logger.info("MEMORY_MANAGER - get_memory_manager called")
    return MemoryManager()
