from io import BytesIO
from openai import OpenAI
from typing import Optional
from ai_companion.settings import settings
import logging

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class SpeechToText:
    """A class to handle speech-to-text conversion using Groq's Whisper model."""

    # Required environment variables
    REQUIRED_ENV_VARS = ["GROQ_API_KEY"]

    def __init__(self):
        # --- ElevenLabs STT Configuration ---
        self.api_key = settings.OPENAI_API_KEY # Use your ElevenLabs STT API Key
        self.model = settings.STT_MODEL_NAME # ElevenLabs Scribe model ID (check documentation for exact name if different)
        self.base_url = settings.OPENAI_BASE_URL
        self.transcription_url = f"{self.base_url}/audio/transcriptions"

    async def transcribe(self, audio_data: bytes, mime_type: Optional[str] = None) -> str:
        
        if not audio_data:
            logger.error("No audio data provided for transcription.")
            raise ValueError("Audio data cannot be empty.")
        
        # Determine filename based on mime_type
        filename = "audio.wav" # Default
        if mime_type:
            if mime_type == "audio/mpeg":
                filename = "audio.mp3"
            elif mime_type == "audio/wav":
                filename = "audio.wav"
            elif mime_type == "audio/webm": # Keep WebM support
                filename = "audio.webm"
            elif mime_type == "audio/x-m4a" or mime_type == "audio/mp4": # Common for M4A/AAC
                filename = "audio.m4a"
            elif mime_type == "audio/flac":
                filename = "audio.flac"
            # Add other types as needed
            else:
                logger.warning(f"Unhandled mime_type: {mime_type}. Defaulting to .wav filename.")

        audio_file = BytesIO(audio_data)
        audio_file.name = filename  # Set the filename for the BytesIO object
        logger.info(f"Sending audio for transcription to OpenAI (model: {self.model}, type: {mime_type}, filename: {filename})...")
        
        logger.info(f"Sending audio for transcription to OpenAI (model: {self.model}, type: {mime_type}, filename: {filename})...")

        with OpenAI(api_key=self.api_key, base_url=self.base_url) as client:
            transcription = client.audio.transcriptions.create(
                file=audio_file,
                model=self.model, # OpenAI Whisper API expects 'text' response format
            )

            if not transcription:
                 logger.error(f"OpenAI transcription response missing 'text' key: {transcription}")
                 raise Exception("OpenAI transcription failed to return text.")

            logger.info(f"Transcription received from OpenAI: {transcription}...")
            return transcription
