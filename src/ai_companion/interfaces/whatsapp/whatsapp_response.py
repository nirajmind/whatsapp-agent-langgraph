import logging
import os
from io import BytesIO
from typing import Dict, Optional, Any
from contextlib import asynccontextmanager # Import asynccontextmanager
import httpx
import uuid # Import uuid for per-request IDs

from fastapi import APIRouter, Request, Response, HTTPException # Import HTTPException
from fastapi.responses import PlainTextResponse

# MongoDB Async Client (ensure this is from pymongo.mongodb_asyncio)
from pymongo import MongoClient as AsyncMongoClient
from langgraph.checkpoint.mongodb.aio import AsyncMongoDBSaver

from langchain_core.messages import HumanMessage # Assuming HumanMessage is needed here
from ai_companion.graph import graph_builder
from ai_companion.modules.image import ImageToText
from ai_companion.modules.speech import SpeechToText, TextToSpeech
from ai_companion.settings import settings

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# --- Global Module Instances ---
# Initialize these globally once since they're stateless or manage internal state
speech_to_text = SpeechToText()
text_to_speech = TextToSpeech()
image_to_text = ImageToText()

# Router for WhatsApp respo
whatsapp_router = APIRouter()

# WhatsApp API credentials
WHATSAPP_TOKEN = os.getenv("WHATSAPP_TOKEN")
WHATSAPP_PHONE_NUMBER_ID = os.getenv("WHATSAPP_PHONE_NUMBER_ID")

# --- FastAPI App Initialization (Pass lifespan) ---
# If this file is the root FastAPI app, use FastAPI().
# If this is a sub-router, you'd define lifespan on the main app where this router is included.
whatsapp_router = APIRouter() # Pass the lifespan context manager

# --- Helper functions for WhatsApp Media (moved outside main handler) ---
async def download_media(media_id: str) -> bytes:
    """Download media from WhatsApp."""
    media_metadata_url = f"https://graph.facebook.com/v22.0/{media_id}"
    headers = {"Authorization": f"Bearer {WHATSAPP_TOKEN}"}
    logger.info(f"Downloading media metadata from {media_metadata_url}")
    async with httpx.AsyncClient() as client:
        metadata_response = await client.get(media_metadata_url, headers=headers)
        metadata_response.raise_for_status()
        metadata = metadata_response.json()
        download_url = metadata.get("url")
        logger.info(f"Downloading media content from {download_url}")
        media_response = await client.get(download_url, headers=headers)
        media_response.raise_for_status()
        return media_response.content


async def process_audio_message(message: Dict) -> str:
    """Download and transcribe audio message."""
    audio_id = message["audio"]["id"]
    media_metadata_url = f"https://graph.facebook.com/v22.0/{audio_id}"
    headers = {"Authorization": f"Bearer {WHATSAPP_TOKEN}"}
    logger.info(f"Processing audio message with ID: {audio_id}")
    async with httpx.AsyncClient() as client:
        metadata_response = await client.get(media_metadata_url, headers=headers)
        metadata_response.raise_for_status()
        metadata = metadata_response.json()
        download_url = metadata.get("url")

    # Download the audio file
    async with httpx.AsyncClient() as client:
        audio_response = await client.get(download_url, headers=headers)
        audio_response.raise_for_status()

    # Prepare for transcription
    audio_buffer = BytesIO(audio_response.content)
    audio_buffer.seek(0)
    audio_data = audio_buffer.read()
    
    logger.info("Transcribing audio content...")
    return await speech_to_text.transcribe(audio_data)


async def upload_media(media_content: BytesIO, mime_type: str) -> str:
    """Upload media to WhatsApp servers."""
    headers = {"Authorization": f"Bearer {WHATSAPP_TOKEN}"}
    files = {"file": ("response_media", media_content, mime_type)} # Changed filename for consistency
    data = {"messaging_product": "whatsapp", "type": mime_type}
    logger.info(f"Uploading media of type: {mime_type} contnent length: {len(media_content.getbuffer())} bytes with whatsapp phone number ID: {WHATSAPP_PHONE_NUMBER_ID}")
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"https://graph.facebook.com/v22.0/{WHATSAPP_PHONE_NUMBER_ID}/media",
            headers=headers,
            files=files,
            data=data,
        )
        response.raise_for_status() # Raise for HTTP errors
        result = response.json()
    
    if "id" not in result:
        logger.error(f"Media upload failed: {result}")
        raise Exception("Failed to upload media, no ID received.")
    logger.info(f"Media uploaded with ID: {result['id']}")
    return result["id"]


async def send_response(
    from_number: str,
    response_text: str,
    message_type: str = "text",
    media_content: Optional[bytes] = None,
) -> bool:
    """Send response to user via WhatsApp API."""
    headers = {
        "Authorization": f"Bearer {WHATSAPP_TOKEN}",
        "Content-Type": "application/json",
    }
    json_data: Dict[str, Any] = {} # Initialize to allow flexible assignment

    if message_type in ["audio", "image"] and media_content:
        try:
            mime_type = "audio/mpeg" if message_type == "audio" else "image/png"
            media_buffer = BytesIO(media_content)
            media_id = await upload_media(media_buffer, mime_type)
            json_data = {
                "messaging_product": "whatsapp",
                "to": from_number,
                "type": message_type,
                message_type: {"id": media_id},
            }

            # Add caption for images
            if message_type == "image":
                json_data["image"]["caption"] = response_text
            logger.info(f"Sending {message_type} response to {from_number} with media ID: {media_id} and payload: {json_data}")
        except Exception as e:
            logger.error(f"Media upload failed for {message_type}, falling back to text: {e}", exc_info=True)
            message_type = "text" # Fallback if media upload fails

    if message_type == "text":
        json_data = {
            "messaging_product": "whatsapp",
            "to": from_number,
            "type": "text",
            "text": {"body": response_text},
        }
        logger.info(f"Sending text response to {from_number}: {response_text}")

    # Remove print statements for production code
    # print(headers)
    # print(json_data)

    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"https://graph.facebook.com/v22.0/{WHATSAPP_PHONE_NUMBER_ID}/messages",
            headers=headers,
            json=json_data,
        )
    response.raise_for_status() # Raise for HTTP errors
    logger.info(f"WhatsApp API response status: {response.status_code}")
    return response.status_code == 200


# --- WhatsApp Webhook Endpoint ---
@whatsapp_router.api_route("/whatsapp_response", methods=["GET", "POST"]) # Using 'app' as the router
async def whatsapp_handler(request: Request) -> Response:
    # --- CRITICAL FIX: Per-request UUID ---
    request_id = str(uuid.uuid4())
    logger.info(f"[{request_id}] Received webhook request from {request.client.host}")
    # --- END CRITICAL FIX ---

    if request.method == "GET":
        params = request.query_params
        if params.get("hub.verify_token") == os.getenv("WHATSAPP_VERIFY_TOKEN"):
            logger.info(f"[{request_id}] Webhook verification successful.")
            return Response(content=params.get("hub.challenge"), status_code=200)
        logger.warning(f"[{request_id}] Webhook verification failed: Token mismatch.")
        return Response(content="Verification token mismatch", status_code=403)

    try:
        data = await request.json()
        logger.debug(f"[{request_id}] Incoming webhook data: {data}") # Log full incoming data for debugging
        
        # Ensure data structure is as expected from WhatsApp Cloud API
        if not (isinstance(data, dict) and "entry" in data and len(data["entry"]) > 0 and
                "changes" in data["entry"][0] and len(data["entry"][0]["changes"]) > 0 and
                "value" in data["entry"][0]["changes"][0]):
            logger.warning(f"[{request_id}] Unexpected webhook data structure: {data}")
            return Response(content="Invalid data structure", status_code=400)

        change_value = data["entry"][0]["changes"][0]["value"]
        
        # Handle message events
        if "messages" in change_value and len(change_value["messages"]) > 0:
            message = change_value["messages"][0]
            from_number = message["from"] # This is the unique user ID from WhatsApp
            session_id = from_number # Use WhatsApp user ID as session_id/thread_id
            
            logger.info(f"[{request_id}] Message from {from_number}, type: {message.get('type')}")

            # Determine message content
            content = ""
            if message["type"] == "audio":
                content = await process_audio_message(message)
            elif message["type"] == "image":
                content = message.get("image", {}).get("caption", "")
                image_bytes = await download_media(message["image"]["id"])
                try:
                    description = await image_to_text.analyze_image(
                        image_bytes,
                        "Please describe what you see in this image in the context of our conversation.",
                    )
                    content += f"\n[Image Analysis: {description}]"
                    logger.info(f"[{request_id}] Image analysis completed: {description}")
                except Exception as e:
                    logger.warning(f"[{request_id}] Failed to analyze image: {e}")
            elif message["type"] == "text":
                content = message["text"]["body"]
            else:
                logger.warning(f"[{request_id}] Unhandled message type: {message['type']}")
                return Response(content="Unhandled message type", status_code=200) # Still return 200 to WhatsApp

            logger.info(f"[{request_id}] Message content: {content}")

            # --- CRITICAL FIX: Load previous state to get user_name ---
            # --- CRITICAL FIX: Get checkpointer from app.state ---
            if not hasattr(request.app.state, "global_checkpointer") or request.app.state.global_checkpointer is None:
                logger.error(f"[{request_id}] Global checkpointer not found in app.state. Cannot process message.")
                raise HTTPException(status_code=503, detail="Service not ready: Database connection failed.")
            
            checkpointer_to_use = request.app.state.global_checkpointer
            # --- END CRITICAL FIX ---

            previous_state_snapshot_tuple = await checkpointer_to_use.aget_tuple(
                config={"configurable": {"thread_id": session_id}} # Use session_id as thread_id
            )
            
            previous_state_values = {}
            if previous_state_snapshot_tuple:
                previous_state_values = previous_state_snapshot_tuple[1]
                logger.info(f"[{request_id}] Loaded previous state. Values: {previous_state_values}")
            else:
                logger.info(f"[{request_id}] No previous state found for thread_id: {session_id}")
            
            # Determine the user_name for this turn's config
            # Prioritize from previous state, else use WhatsApp user ID (from_number) as initial user_name
            current_user_name_for_config = previous_state_values.get("user_name", from_number)
            
            # Pass the current user_name in the config for this invocation
            config = {"configurable": {"thread_id": session_id, "user_name": current_user_name_for_config}}
            # --- END CRITICAL FIX ---

            # Process message through the graph agent
            graph = graph_builder.compile(checkpointer=checkpointer_to_use)
            logger.info(f"[{request_id}] Starting graph processing for thread: {session_id}")
            
            # LangGraph ainvoke will persist state based on the checkpointer and config
            output_state = await graph.ainvoke( # Using ainvoke here for non-streaming WhatsApp
                {"messages": [HumanMessage(content=content)]},
                config,
            )
            logger.info(f"[{request_id}] Graph processing completed. Output state: {output_state}")

            # Get the workflow type and response from the state
            # Ensure output_state is not None before accessing values
            response_message = ""
            workflow = "conversation"
            audio_buffer = None
            image_path = None

            if output_state and output_state.values.get("messages"):
                final_ai_message = output_state.values["messages"][-1]
                if hasattr(final_ai_message, 'content'):
                    response_message = final_ai_message.content
                workflow = output_state.values.get("workflow", "conversation")
                if workflow == "audio":
                    audio_buffer = output_state.values.get("audio_buffer")
                elif workflow == "image":
                    image_path = output_state.values.get("image_path")

            logger.info(f"[{request_id}] Final response message from state: '{response_message}' (Workflow: {workflow})")

            # Handle different response types based on workflow
            success = False
            if workflow == "audio" and audio_buffer:
                success = await send_response(from_number, response_message, "audio", audio_buffer)
            elif workflow == "image" and image_path:
                try:
                    with open(image_path, "rb") as f:
                        image_data = f.read()
                    success = await send_response(from_number, response_message, "image", image_data)
                except FileNotFoundError:
                    logger.error(f"[{request_id}] Image file not found at path {image_path}. Falling back to text.")
                    success = await send_response(from_number, response_message, "text") # Fallback
                except Exception as e:
                    logger.error(f"[{request_id}] Error loading image from {image_path}: {e}", exc_info=True)
                    success = await send_response(from_number, response_message, "text") # Fallback
            else: # Default text conversation
                success = await send_response(from_number, response_message, "text")

            if not success:
                logger.error(f"[{request_id}] Failed to send final message to {from_number}.")
                # Depending on requirements, you might send an error message to the user here
                return Response(content="Failed to send message", status_code=500)

            logger.info(f"[{request_id}] Message processed successfully for {from_number}.")
            return Response(content="Message processed", status_code=200)

        # Handle status updates
        elif "statuses" in change_value:
            logger.info(f"[{request_id}] Status update received: {change_value.get('statuses', [])}")
            return Response(content="Status update received", status_code=200)

        else:
            logger.warning(f"[{request_id}] Unknown event type in webhook: {data}")
            return Response(content="Unknown event type", status_code=400)

    except Exception as e:
        logger.error(f"[{request_id}] Error processing webhook: {e}", exc_info=True)
        return PlainTextResponse("Internal server error", status_code=500)