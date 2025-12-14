import chainlit as cl
from langchain_core.messages import AIMessageChunk, HumanMessage
from langgraph.checkpoint.mongodb.aio import AsyncMongoDBSaver # <--- CORRECT for v0.3.x
from pymongo import AsyncMongoClient # Already imported, good.
import uuid

from sympy import true

from ai_companion.graph import graph_builder
from ai_companion.interfaces.whatsapp.whatsapp_response import send_response
from ai_companion.modules.image import ImageToText
from ai_companion.modules.speech import SpeechToText, TextToSpeech
from ai_companion.settings import settings
from io import BytesIO
from typing import Any, Optional

# In ai_companion/interfaces/chainlit/app.py
import logging # Already imported, but ensure used.
logger = logging.getLogger(__name__) # Ensure module-level logger

print("DEBUG: Executing app.py script content starting...") # Raw print, appears very early
logger.info("DEBUG: app.py is starting execution via Chainlit logger.") # Via Chainlit's logger

# Global module instances
speech_to_text = SpeechToText()
text_to_speech = TextToSpeech()
image_to_text = ImageToText()

# --- CRITICAL FIX: Initialize Checkpointer GLOBALLY and keep it open ---
# This will call get_vector_store() via MemoryManager, which initializes its MongoClient
# But for the checkpointing, we need its own instance
global_mongo_client: Optional[AsyncMongoClient] = None
global_checkpointer: Optional[AsyncMongoDBSaver] = None

# --- FIX: Corrected decorator and function definition ---
async def initialize_global_db_clients():
    """Initializes the global MongoDB client and LangGraph checkpointer."""
    global global_mongo_client, global_checkpointer
    if global_mongo_client is None: # Check if already initialized
        try:
            # Initialize the global MongoDB client (AsyncMongoClient)
            global_mongo_client = AsyncMongoClient(settings.MONGO_URI)
            cl.logger.info("APP - Global MongoDB client initialized.")

            # Initialize the global checkpointer with the existing client
            global_checkpointer = AsyncMongoDBSaver(
                client=global_mongo_client,
                database_name=settings.DATABASE_NAME,
                collection_name=settings.COLLECTION_NAME
            )
            cl.logger.info("APP - Global checkpointer initialized with shared client.")

        except Exception as e:
            cl.logger.error(f"APP - Critical: Failed to initialize global MongoDB client or checkpointer: {e}", exc_info=True)
            # Re-raise the exception to prevent the app from starting with a broken DB connection
            raise

@cl.on_chat_start
async def on_chat_start():
    """Initialize the chat session with a unique thread ID and ensure global DB clients are ready."""
    # --- CRITICAL FIX: Call the global initialization function here ---
    await initialize_global_db_clients()
    # --- END CRITICAL FIX ---
    session_id = str(uuid.uuid4())
    cl.user_session.set("session_id", session_id)
    thread_id = cl.user_session.get("thread_id", session_id)  # Use session_id as fallback
    cl.user_session.set("thread_id", thread_id)
    cl.logger.info(f"APP - Chat started with thread_id: {thread_id}")

@cl.on_message
async def on_message(message: cl.Message):
    """Handle text messages and images"""
    try:
        cl.logger.info(f"Starting the CHAINLIT message handler. Received message: {message.content}")
        msg = cl.Message(content="")
        await msg.send()  # Send empty message to start streaming

        # Process any attached images
        content = message.content
        if message.elements:
            cl.logger.info(f"Message has {len(message.elements)} elements.")
            for elem in message.elements:
                if isinstance(elem, cl.Image):
                    cl.logger.info("Processing attached image")
                    # Read image file content
                    with open(elem.path, "rb") as f:
                        image_bytes = f.read()

                    # Analyze image and add to message content
                    try:
                        # Use global ImageToText instance
                        description = await image_to_text.analyze_image(
                            image_bytes,
                            "Please describe what you see in this image in the context of our conversation.",
                        )
                        # Add a marker for the image analysis in the content
                        image_counter = cl.user_session.get("image_counter", 0) + 1
                        cl.user_session.set("image_counter", image_counter)
                        image_marker = f"[User uploaded image {image_counter}]"
                        content += f"\n{image_marker}\n[Image Analysis: {description}]"
                        cl.logger.info(f"Image analysis completed: {description}")
                        # Optionally display the image analysis as a separate step/message
                        # await cl.Message(content=f"Image Analysis: {description}", parent_id=message.id).send()
                    except Exception as e:
                        cl.logger.error(f"Failed to analyze image: {e}")
                        content += "\n[Error analyzing image]"

        # Process through graph with enriched message content
        thread_id = cl.user_session.get("thread_id")
        if not thread_id:
             cl.logger.error("Thread ID not found in session!")
             # Handle error: maybe re-initialize or send an error message
             await cl.Message(content="Error: Session expired or invalid. Please refresh.").send()
             return
        cl.logger.info(f"Processing message with thread_id: {thread_id}")

        # --- Checkpointer Change: Use AsyncMongoDBSaver ---
        # Ensure you have MONGODB_URI, MONGODB_CHECKPOINT_DB_NAME,
        # and MONGODB_CHECKPOINT_COLLECTION_NAME defined in your settings.py
        # --- CRITICAL FIX: Use async with to get checkpointer instance ---
        if not all([settings.MONGO_URI, settings.DATABASE_NAME, settings.COLLECTION_NAME]):
            cl.logger.error("MongoDB connection details are missing in settings.")
            raise ValueError("MongoDB connection details (URI, DB Name, Collection Name) are not configured in settings.")

        # --- CRITICAL FIX: Ensure global checkpointer is available ---
        if global_checkpointer is None:
            cl.logger.error("APP - Global checkpointer not available during on_message. Re-initializing (should not happen often).")
            await initialize_global_db_clients() # Attempt re-init if somehow not ready
            if global_checkpointer is None: # If re-init also fails
                await cl.Message(content="Error: Application not fully initialized. Please restart.").send()
                return
        # --- END CRITICAL FIX ---

        # Load previous state using the global checkpointer
        previous_state_snapshot_tuple = await global_checkpointer.aget_tuple(
            config={"configurable": {"thread_id": thread_id}}
        )
        previous_state_values = {}
        user_name_from_persisted_state = None
        if previous_state_snapshot_tuple:
            previous_state_values = previous_state_snapshot_tuple[1]
            # --- CRITICAL FIX: Correctly access nested user_name ---
        if previous_state_values:
            # Try to get from the top level first (where it's returned by router/memory_injection)
            user_name_from_persisted_state = previous_state_values.get('user_name')

            if user_name_from_persisted_state is None:
                # If not at the top level, look deeper into channel_values.__root__
                channel_values = previous_state_values.get('channel_values', {})
                root_channel_values = channel_values.get('__root__', {})
                if isinstance(root_channel_values, dict): # Ensure it's a dict before accessing
                    user_name_from_persisted_state = root_channel_values.get('user_name')

        # Prioritize the name from persisted state, else use thread_id
        current_user_name_for_config = user_name_from_persisted_state if user_name_from_persisted_state else thread_id
        # --- END CRITICAL FIX ---

        cl.logger.info(f"APP - Derived user_name for config: {current_user_name_for_config}")

        config = {"configurable": {"thread_id": thread_id, "user_name": current_user_name_for_config}}
                # --- END CRITICAL FIX ---

        async with cl.Step(type="run"):
            # The checkpointer is already initialized above.
            # No need for 'async with AsyncMongoDBSaver.from_conn_string(...)' here again.
            graph = graph_builder.compile(checkpointer=global_checkpointer)
            cl.logger.info("Starting graph processing")
            config = {"configurable": {"thread_id": thread_id, "user_name": current_user_name_for_config}}
            cl.logger.info(f"APP - on_message - thread_id: {thread_id}, passing config: {config}")
            received_chunks = False

            async for chunk in graph.astream(
                {"messages": [HumanMessage(content=content)]},
                config,
                # Use the thread_id as a unique identifier for the session if the user_name is not available
                stream_mode="messages",
                ):
                    # cl.logger.info(f"Streaming messages, received chunk: {chunk}")
                    if isinstance(chunk[0], AIMessageChunk):
                        received_chunks = True
                        # cl.logger.info(f"Streaming token: {chunk[0].content}")
                        await msg.stream_token(chunk[0].content)

            output_state = await graph.aget_state(config=config)
            # cl.logger.info(f"[{thread_id}] Graph processing completed. Output state: {output_state}")

            final_response_content = ""
            final_elements = []

            # --- CRITICAL FIX: Determine workflow based on presence of content/elements ---
            response_message_from_state = ""
            audio_buffer_from_state = None
            image_path_from_state = None
            final_message_content_from_state = ""

            if output_state and output_state.values.get("messages"):
                # Get the content of the *last* message in the state
                final_message_from_state = output_state.values["messages"][-1]
                final_response_content = (final_message_from_state.content if hasattr(final_message_from_state, 'Ava') or hasattr(final_message_from_state, 'content') else final_message_from_state)
                cl.logger.info(f"Final response message from state: {final_response_content}")
                cl.logger.info(f"Final message from state: {final_message_from_state}")
                if isinstance(final_message_from_state, AIMessageChunk): # Should be AIMessage usually
                     final_message_content_from_state = final_message_from_state.content
                elif hasattr(final_message_from_state, 'content'):
                    final_message_content_from_state = final_message_from_state.content
                    audio_buffer_from_state = output_state.values.get("audio_buffer")
                    image_path_from_state = output_state.values.get("image_path")
                elif isinstance(final_message_from_state, bytes):
                    final_message_content_from_state = output_state.values["messages"][-2].content
                    audio_buffer_from_state = final_message_from_state

                cl.logger.info(f"APP - Final response from LLM: '{final_message_content_from_state}'")
                cl.logger.info(f"APP - Audio buffer present: {audio_buffer_from_state is not None}, Image path present: {image_path_from_state is not None}")
            
        success = False
        if audio_buffer_from_state:
            cl.logger.info("APP - Sending audio response workflow.")
            audio_element = cl.Audio(
                name="AI Audio Response",
                auto_play=True, # Consider making this configurable or False
                mime="audio/mpeg", # Adjust if your TTS outputs a different format
                content=audio_buffer_from_state,
                display="inline"
            )
            success = await send_response("971543108449", response_message_from_state, "audio", audio_buffer_from_state)
            if not success:
                cl.logger.error("Failed to send audio response, falling back to text.")
                success = await send_response(None, response_message_from_state, "text")
            else:
                cl.logger.info("Audio response sent successfully.")
                await cl.Message(content=final_message_content_from_state, elements=[audio_element]).send()
        elif image_path_from_state:
            cl.logger.info("APP - Sending image response workflow.")
            try:
                with open(image_path_from_state, "rb") as f:
                    image_data = f.read()
                    cl.logger.info(f"Loaded image data from {image_path_from_state} ({len(image_data)} bytes)")
                    success = await send_response("971543108449", response_message_from_state, "image", image_data)
                    if not success:
                        cl.logger.error("Failed to send image response, falling back to text.")
                        success = await send_response(None, response_message_from_state, "text")
                    else:
                        cl.logger.info("Image response sent successfully.")
                        image_element = cl.Image(
                            path=image_path_from_state,
                            name="AI Image Response",
                            content=image_data,
                            display="inline"
                        )    
                        await cl.Message(
                            content=final_message_content_from_state,
                            elements=[image_element]
                        ).send()
            except FileNotFoundError:
                cl.logger.error(f"APP - Image file not found at path {image_path_from_state}. Falling back to text.")
                success = await send_response(None, response_message_from_state, "text") # Fallback
            except Exception as e:
                cl.logger.error(f"APP - Error loading image from {image_path_from_state}: {e}", exc_info=True)
                success = await send_response(None, response_message_from_state, "text") # Fallback
        else: # Default text conversation if no audio/image content
            cl.logger.info("APP - Sending text conversation workflow.")
            if final_message_content_from_state:
                # If content was streamed, msg is already partially sent.
                # If not, send a new message. This logic is complex.
                # Simplest for now: always send a new message if not streaming
                if not received_chunks:
                    await cl.Message(content=final_message_content_from_state, elements=final_elements).send()
                else: # If chunks were streamed, msg already has content, just update elements
                    msg.content = final_message_content_from_state # Ensure msg has full content
                    msg.elements = final_elements
                    await msg.update() # Update the existing message
            elif not received_chunks:
                await cl.Message(content="I couldn't generate a response for that.").send()

    except Exception as e:
        cl.logger.error(f"Error in on_message: {e}", exc_info=True) # Log traceback
        await cl.Message(content=f"An error occurred: {str(e)}").send()

@cl.on_audio_chunk
async def on_audio_chunk(chunk: cl.AudioChunk):
    """Handle incoming audio chunks"""
    try:
        if chunk.isStart:
            # Use a temporary file or BytesIO buffer
            buffer = BytesIO()
            # Try to get a reasonable extension, default to 'wav' or 'mp3' if unsure
            mime_type = chunk.mimeType or "audio/wav" # Provide a default
            # Basic parsing, handle potential parameters like '; codecs=opus'
            extension = mime_type.split('/')[-1].split(';')[0].strip()
            # Ensure a valid extension name (e.g., replace 'mpeg' with 'mp3' if common)
            if extension == "mpeg": extension = "mp3"
            if not extension: extension = "wav" # Fallback extension

            # Use a unique identifier if possible, otherwise fallback
            session_id = cl.user_session.get('id', 'unknown_session')
            buffer.name = f"input_audio_{session_id}.{extension}"

            cl.user_session.set("audio_buffer", buffer)
            cl.user_session.set("audio_mime_type", mime_type)
            cl.logger.info(f"Started receiving audio chunk: {mime_type} (filename: {buffer.name})")

        # Append data to the buffer in the session
        audio_buffer = cl.user_session.get("audio_buffer")
        if audio_buffer:
            audio_buffer.write(chunk.data)
        else:
            # This might happen if chunks arrive before isStart or after on_audio_end cleanup
            cl.logger.warning("Received audio chunk but no buffer found in session (isStart might be missed or already ended).")

    except Exception as e:
        cl.logger.error(f"Error in on_audio_chunk: {e}", exc_info=True)


@cl.on_audio_end
async def on_audio_end(elements: Any):
    """Process completed audio input with robust error handling and element type check."""
    audio_buffer = None # Ensure buffer is defined for finally block
    try:
        # --- Runtime Type Check for 'elements' ---
        if not isinstance(elements, list):
            cl.logger.warning(f"on_audio_end received unexpected type for 'elements': {type(elements)}. Expected a list.")
            elements = []  # Initialize as an empty list to prevent errors later
        # --- End of Type Check ---

        audio_buffer = cl.user_session.get("audio_buffer")
        mime_type = cl.user_session.get("audio_mime_type", "audio/wav") # Get mime type

        if not audio_buffer:
            cl.logger.error("Audio processing ended, but no audio buffer found in session.")
            await cl.Message(content="Error: Could not process the received audio (buffer missing).").send()
            return

        # Ensure buffer is ready to be read
        audio_buffer.seek(0)
        audio_data = audio_buffer.read()

        # It's good practice to clear session state once data is read
        cl.user_session.set("audio_buffer", None)
        cl.user_session.set("audio_mime_type", None)

        if not audio_data:
            cl.logger.error("Audio processing ended, but audio data is empty.")
            await cl.Message(content="Error: Received empty audio.").send()
            # Close the buffer even if data is empty
            try:
                audio_buffer.close()
            except Exception: pass # Ignore cleanup errors
            return

        # Show user's audio message
        cl.logger.info(f"Processing received audio ({len(audio_data)} bytes, type: {mime_type})")
        # Use the determined mime_type for the input element
        input_audio_el = cl.Audio(
            mime=mime_type,
            content=audio_data,
            name="Your Audio Input",
            display="inline"
        )
        # Send the user's audio input visualization
        await cl.Message(
            author="You",
            content="", # No text content for the user's audio message itself
            elements=[input_audio_el, *elements] # Include the audio element
        ).send()

        # Transcribe audio
        cl.logger.info("Transcribing audio...")
        # Use global SpeechToText instance
        transcription = await speech_to_text.transcribe(audio_data, mime_type=mime_type)
        cl.logger.info(f"Transcription: {transcription}")

        # Send transcription as a separate message for clarity
        await cl.Message(author="Your (Transcription)", content=transcription).send()

        thread_id = cl.user_session.get("thread_id")
        if not thread_id:
            cl.logger.error("Thread ID not found in session during audio end!")
            await cl.Message(content="Error: Session expired or invalid. Please refresh.").send()
            # Close the buffer before returning
            try:
                audio_buffer.close()
            except Exception: pass # Ignore cleanup errors
            return

        # --- Checkpointer Change: Use AsyncMongoDBSaver ---
        if not all([settings.MONGO_URI, settings.DATABASE_NAME, settings.COLLECTION_NAME]):
            cl.logger.error("MongoDB connection details are missing in settings.")
            raise ValueError("MongoDB connection details (URI, DB Name, Collection Name) are not configured in settings.")

        # --- CRITICAL FIX: Ensure global checkpointer is available ---
        if global_checkpointer is None:
            cl.logger.error("APP - Global checkpointer not available during on_message. Re-initializing (should not happen often).")
            await initialize_global_db_clients() # Attempt re-init if somehow not ready
            if global_checkpointer is None: # If re-init also fails
                await cl.Message(content="Error: Application not fully initialized. Please restart.").send()
                return
        # --- END CRITICAL FIX ---

        # Load previous state using the global checkpointer
        previous_state_snapshot_tuple = await global_checkpointer.aget_tuple(
            config={"configurable": {"thread_id": thread_id}}
        )
        previous_state_values = {}
        user_name_from_persisted_state = None
        if previous_state_snapshot_tuple:
            previous_state_values = previous_state_snapshot_tuple[1]
            # --- CRITICAL FIX: Correctly access nested user_name ---
        if previous_state_values:
            # Try to get from the top level first (where it's returned by router/memory_injection)
            user_name_from_persisted_state = previous_state_values.get('user_name')

            if user_name_from_persisted_state is None:
                # If not at the top level, look deeper into channel_values.__root__
                channel_values = previous_state_values.get('channel_values', {})
                root_channel_values = channel_values.get('__root__', {})
                if isinstance(root_channel_values, dict): # Ensure it's a dict before accessing
                    user_name_from_persisted_state = root_channel_values.get('user_name')

        # Prioritize the name from persisted state, else use thread_id
        current_user_name_for_config = user_name_from_persisted_state if user_name_from_persisted_state else thread_id
        # --- END CRITICAL FIX ---

        cl.logger.info(f"APP - Derived user_name for config: {current_user_name_for_config}")

        config = {"configurable": {"thread_id": thread_id, "user_name": current_user_name_for_config}}
        # --- END CRITICAL FIX ---
        graph = graph_builder.compile(checkpointer=global_checkpointer)

        cl.logger.info(f"Invoking graph with transcription... {transcription.text}")
        config = {"configurable": {"thread_id": thread_id, "user_name": thread_id}}
        cl.logger.info(f"APP - on_audio_end - thread_id: {thread_id}, passing config: {config}")
        # Use ainvoke for potentially long-running graph execution
        output_state = await graph.ainvoke(
            {"messages": [HumanMessage(content=transcription.text)]},
            config,
        )

        # cl.logger.info(f"Graph invocation complete - Output state received. {output_state}")

        # Check for response message and potential audio output
        if output_state and output_state.get("messages"):
            # Get the content of the *last* message
            final_response_message_obj = output_state["messages"][-1]
            final_response_message = final_response_message_obj.content if hasattr(final_response_message_obj, 'content') else ""

            if not final_response_message:
                cl.logger.warning("Graph returned a final message object, but its content is empty.")
                await cl.Message(content="I received your audio, but didn't get a text response back.").send()
                # Close the buffer before returning
                try:
                    audio_buffer.close()
                except Exception: pass # Ignore cleanup errors
                return

            cl.logger.info(f"Final response message from state: {final_response_message}")

            # Synthesize audio response if workflow indicates or by default for audio input
            # Use global TextToSpeech instance
            cl.logger.info("Synthesizing audio response...")
            audio_response_buffer = await text_to_speech.synthesize(final_response_message)
            cl.logger.info("Audio synthesis complete.")

            if not audio_response_buffer:
                cl.logger.error("Text-to-speech synthesis failed or returned empty buffer.")
                # Send only the text response if audio failed
                await cl.Message(content=final_response_message).send()
                # Close the buffer before returning
                try:
                    audio_buffer.close()
                except Exception: pass # Ignore cleanup errors
                return

            output_audio_el = cl.Audio(
                name="AI Audio Response",
                auto_play=True, # Consider making this configurable or False
                mime="audio/mpeg", # Adjust if your TTS outputs a different format
                content=audio_response_buffer,
                display="inline"
            )
            await cl.Message(
                content=final_response_message,
                elements=[output_audio_el]
            ).send()
        else:
            cl.logger.warning("Graph did not return messages in the output state.")
            await cl.Message(content="I received your audio, but couldn't generate a response.").send()

        # Close the original input buffer now that we're done
        try:
            audio_buffer.close()
        except Exception as close_err:
            cl.logger.warning(f"Could not close audio input buffer: {close_err}")

    except Exception as e:
        cl.logger.error(f"Error in on_audio_end: {e}", exc_info=True)
        await cl.Message(content=f"An error occurred while processing your audio: {str(e)}").send()
    finally:
        # Ensure buffer is cleaned up from session and closed if it still exists
        # (e.g., if an error occurred before it was explicitly closed)
        final_check_buffer = cl.user_session.get("audio_buffer")
        if final_check_buffer:
            try:
                final_check_buffer.close()
            except Exception:
                pass # Ignore errors during final cleanup
            cl.user_session.set("audio_buffer", None)
        # Also ensure mime type is cleared
        if cl.user_session.get("audio_mime_type"):
            cl.user_session.set("audio_mime_type", None)

