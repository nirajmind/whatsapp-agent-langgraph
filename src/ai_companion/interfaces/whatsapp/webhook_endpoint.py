from contextlib import asynccontextmanager
import logging
from typing import Optional
from fastapi import FastAPI, HTTPException
from pymongo import MongoClient as AsyncMongoClient
from langgraph.checkpoint.mongodb.aio import AsyncMongoDBSaver

from ai_companion.settings import settings
from ai_companion.interfaces.whatsapp.whatsapp_response import whatsapp_router

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):

    try:
        global_mongo_client = AsyncMongoClient(settings.MONGO_URI)
        logger.info("APP - Global MongoDB client initialized.")

        global_checkpointer = AsyncMongoDBSaver(
            client=global_mongo_client,
            database_name=settings.DATABASE_NAME,
            collection_name=settings.COLLECTION_NAME
        )
        logger.info("APP - Global checkpointer initialized with shared client.")

        # --- CRITICAL FIX: Store the checkpointer in app.state ---
        app.state.global_checkpointer = global_checkpointer
        # --- END CRITICAL FIX ---
    except Exception as e:
        logger.error(f"APP - Critical: Failed to initialize global MongoDB client or checkpointer: {e}", exc_info=True)
        # In FastAPI startup, raising an exception will prevent the app from starting.
        raise HTTPException(status_code=500, detail=f"Failed to initialize database: {e}")
    
    yield # This yields control to the application, which will run until shutdown

    # --- Shutdown Logic (formerly @app.on_event("shutdown")) ---
    if hasattr(app.state, "global_checkpointer") and hasattr(app.state.global_checkpointer, "client"):
        # This is a safe way to close the underlying Motor client
        if app.state.global_checkpointer.client:
            app.state.global_checkpointer.client.close()
            logger.info("APP - Global MongoDB client closed via checkpointer.")

app = FastAPI(lifespan=lifespan, title="AI Companion WhatsApp Interface", version="1.0.0")
app.include_router(whatsapp_router)
