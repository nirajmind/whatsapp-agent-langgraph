from calendar import c
import os
import logging
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from threading import local
from typing import List, Optional
import chainlit as cl
from pymongo import AsyncMongoClient
import asyncio

from ai_companion.settings import settings
from sentence_transformers import SentenceTransformer
logger = logging.getLogger(__name__) # Use module-level logger

@dataclass
class Memory:
    """Represents a memory entry in the vector store."""

    text: str
    metadata: dict
    score: Optional[float] = None

    @property
    def id(self) -> Optional[str]:
        return self.metadata.get("id")

    @property
    def timestamp(self) -> Optional[datetime]:
        ts = self.metadata.get("timestamp")
        return datetime.fromisoformat(ts) if ts else None


class VectorStore:
    """A class to handle vector storage operations using MongoDB."""

    REQUIRED_ENV_VARS = ["MONGO_URI", "DATABASE_NAME", "COLLECTION_NAME"]
    EMBEDDING_MODEL = settings.EMBEDDER_NAME  # Use the embedding model from settings
    SIMILARITY_THRESHOLD = 0.7  # Threshold for considering memories as similar

    _instance: Optional["VectorStore"] = None
    _is_actually_initialized: bool = False

    def __new__(cls) -> "VectorStore":
        if cls._instance is None:
            cl.logger.info(f"VECTOR_STORE - __new__ called for VectorStore")
            cls._instance = super().__new__(cls)
            #cls._instance._one_time_init()
        else:
            cl.logger.info(f"VECTOR_STORE - __new__ called for existing VectorStore instance")    
        return cls._instance

    def __init__(self) -> None:
        if not self._is_actually_initialized:
            logger.info(f"VECTOR_STORE - __init__ called for VectorStore (performing actual init).")
            hf_cache_path = os.getenv("HF_HOME", "/hf-sentence-transformers")  # Default path if not set
            # Perform expensive setup here only once
            self._validate_env_vars()
            logger.info(f"VECTOR_STORE - Environment variables validated for actual init with transformer's name '{self.EMBEDDING_MODEL}'.")

            self.model = SentenceTransformer(self.EMBEDDING_MODEL, local_files_only=False)
            logger.info(f"VECTOR_STORE - Embedding model '{self.EMBEDDING_MODEL}' loaded.")

            try: # Add a try-except around client connection for robustness
                self.client = AsyncMongoClient(settings.MONGO_URI)
                self.db = self.client.get_database(settings.DATABASE_NAME)
                self.collection = self.db.get_collection(settings.COLLECTION_NAME)
                self.collection_name = settings.COLLECTION_NAME
                self.vector_index_name = "vector_index" # Match your Atlas Search Index name
                logger.info("VECTOR_STORE - MongoDB client and collection initialized.")
            except Exception as e:
                logger.error(f"VECTOR_STORE - Critical: Failed to connect to MongoDB: {e}", exc_info=True)
                # If connection fails, ensure self.client, self.db, self.collection are None
                # or raise error to prevent partial initialization
                raise # Re-raise if initialization is critical

            self._is_actually_initialized = True # Mark as initialized
        else:
            logger.info("VECTOR_STORE - __init__ called for existing VectorStore instance (skipping re-init).")

    def _validate_env_vars(self) -> None:
        """Validate that all required environment variables are set."""
        missing_vars = [var for var in self.REQUIRED_ENV_VARS if not os.getenv(var)]
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

    async def _run_sync_in_thread(self, func, *args, **kwargs):
        """Helper to run synchronous (blocking) functions in a separate thread."""
        return await asyncio.to_thread(func, *args, **kwargs)    

    async def find_similar_memory(self, text: str) -> Optional[Memory]:
        """Find if a similar memory already exists.

        Args:
            text: The text to search for

        Returns:
            Optional Memory if a similar one is found
        """
        cl.logger.info(f"VECTOR_STORE - find_similar_memory called with text: {text}")
        results = await self.search_memories(text, k=1)
        cl.logger.info(f"VECTOR_STORE - search_memories returned {len(results)} results.")
        if results and results[0].score >= self.SIMILARITY_THRESHOLD:
            cl.logger.info(f"VECTOR_STORE - Similar memory found: {results[0].text} with score: {results[0].score}")
            return results[0]
        cl.logger.info(f"VECTOR_STORE - No similar memory found")
        return None

    async def store_memory(self, text: str, metadata: dict) -> None:
        """Store a new memory in the vector store or update if similar exists.

        Args:
            text: The text content of the memory
            metadata: Additional information about the memory (timestamp, type, etc.)
        """
        # Check if similar memory exists
        cl.logger.info(f"VECTOR_STORE - store_memory called with text: {text} and metadata: {metadata}")
        similar_memory = await self.find_similar_memory(text)
        cl.logger.info(f"VECTOR_STORE - found similar memory - {similar_memory}")
        if similar_memory:
            cl.logger.info(f"VECTOR_STORE - Similar memory found: {similar_memory.text} with score: {similar_memory.score}")
            if similar_memory.id:
                metadata["id"] = similar_memory.id  # Keep same ID for update

        embedding = self.model.encode(text)
        document = {
            "id": metadata.get("id", hash(text)),
            "text": text,
            "vector": embedding.tolist(),
            **metadata
        }

        if similar_memory:
            if similar_memory.id:
                cl.logger.info(f"VECTOR_STORE - Updating existing memory with ID: {similar_memory.id}")
                await self.collection.update_one(
                {"id": similar_memory.id},
                {"$set": document}
            )
        else:
            cl.logger.info(f"VECTOR_STORE - Inserting new memory into - {self.collection_name}")
            try:
                result = await self.collection.insert_one(document) 
                cl.logger.info(f"VECTOR_STORE - Document inserted with ID: {result.inserted_id}")
            except Exception as e:
                cl.logger.error(f"VECTOR_STORE - Error inserting document: {e}", exc_info=True)
                raise

    async def search_memories(self, query: str, k: int = 1) -> List[Memory]:
        """Search for similar memories in the vector store.

        Args:
            query: Text to search for
            k: Number of results to return

        Returns:
            List of Memory objects
        """
        cl.logger.info(f"VECTOR_STORE - search_memories called with query: '{query}' and k={k}")
        if self.collection is None:
            logger.error(f"VECTOR_STORE - search_memories called but MongoDB collection is None for query: '{query}'. Check VectorStore initialization.")
            return [] # Return empty list gracefully if not initialized
        
        try:
            query_embedding = self.model.encode(query, convert_to_tensor=False).tolist()
            pipeline = [
                {
                    "$vectorSearch": {
                        "index": self.vector_index_name,
                        "path": "embedding",
                        "queryVector": query_embedding,
                        "numCandidates": k * 10,
                        "limit": k
                    }
                },
                {
                    "$project": {
                        "text": 1,
                        "metadata": 1,
                        "score": { "$meta": "vectorSearchScore" }
                    }
                }
            ]
            
            logger.info(f"VECTOR_STORE - Executing vector search for '{query}' with k={k}")
            results = []

            # --- CRITICAL FIX: Run synchronous aggregate and iterate over its synchronous cursor ---
            # Use functools.partial to pass the method and arguments to to_thread
            cursor = await self.collection.aggregate(pipeline)
            
            # --- CRITICAL CHANGE: Use async for loop for CommandCursor ---
            async for doc in cursor:
                results.append(Memory(
                    text=doc.get("text"),
                    metadata=doc.get("metadata", {}),
                    score=doc.get("score", 0.0)
                ))
            # --- END CRITICAL CHANGE ---

            logger.info(f"VECTOR_STORE - Found {len(results)} search results.")
            return results
        except Exception as e:
            logger.error(f"VECTOR_STORE - Error searching memories for '{query}': {e}", exc_info=True)
            return []

@lru_cache(maxsize=1)
def get_vector_store() -> VectorStore:
    """Get or create the VectorStore singleton instance."""
    cl.logger.info(f"MEMORY_MANAGER - get_vector_store called")
    return VectorStore()
