import base64
import logging
import os
import httpx
from typing import Optional, List, Dict, Any
from uuid import uuid4 # Import uuid4 directly for convenience

from langchain.prompts import PromptTemplate # Assuming you use this for prompts internally
from pydantic import BaseModel, Field
from together import Together # For Together AI client

from ai_companion.core.exceptions import TextToImageError
from ai_companion.core.prompts import IMAGE_ENHANCEMENT_PROMPT, IMAGE_SCENARIO_PROMPT
from ai_companion.settings import settings # For API keys and model names

logger = logging.getLogger(__name__)

# --- Pydantic Models for Scenario and Enhanced Prompt ---
class ScenarioPrompt(BaseModel):
    """Class for the scenario response"""
    narrative: str = Field(..., description="The AI's narrative response to the question")
    image_prompt: str = Field(..., description="The visual prompt to generate an image representing the scene")

class EnhancedPrompt(BaseModel):
    """Class for the text prompt"""
    content: str = Field(..., description="The enhanced text prompt to generate an image")

# --- TextToImage Class ---
class TextToImage:
    """A class to handle text-to-image generation using Together AI."""

    # Assuming these are the primary env vars needed for this module's functions
    REQUIRED_ENV_VARS = ["GROQ_API_KEY", "TOGETHER_API_KEY"] 

    _instance: Optional["TextToImage"] = None # Singleton instance holder
    _is_actually_initialized: bool = False # Flag for one-time initialization

    def __new__(cls) -> "TextToImage":
        if cls._instance is None:
            logger.info("TEXT_TO_IMAGE - Creating new TextToImage instance (singleton).")
            cls._instance = super().__new__(cls)
            cls._instance._one_time_init() # Call dedicated init method for first time
        else:
            logger.info("TEXT_TO_IMAGE - Reusing existing TextToImage instance (singleton).")
        return cls._instance

    def _one_time_init(self) -> None:
        """Initializes components that only need to be set up once."""
        if self._is_actually_initialized:
            return

        self._validate_env_vars()
        logger.info("TEXT_TO_IMAGE - Environment variables validated.")

        # Initialize Together AI client
        self._together_client: Optional[Together] = None # Will be lazily initialized by property

        # Configuration for LLM calls (e.g., for scenario creation, prompt enhancement)
        self.logger = logging.getLogger(__name__)
        self.groq_api_key = settings.GROQ_API_KEY # Use Groq API key for LLM calls (scenario/enhance)
        self.groq_base_url = "https://api.x.ai/v1/chat/completions" # Assuming this is the Groq-compatible endpoint
        self.groq_headers = {
            "Authorization": f"Bearer {self.groq_api_key}",
            "Content-Type": "application/json"
        }
        self.groq_model_name = settings.TEXT_MODEL_NAME # For scenario/enhance prompts

        self._is_actually_initialized = True
        logger.info("TEXT_TO_IMAGE - TextToImage initialized successfully (one-time).")

    def __init__(self) -> None:
        # This __init__ gets called on every instantiation, but _one_time_init
        # handles the actual setup for the singleton.
        pass

    def _validate_env_vars(self) -> None:
        """Validate that all required environment variables are set."""
        missing_vars = [var for var in self.REQUIRED_ENV_VARS if not os.getenv(var)]
        if missing_vars:
            logger.error(f"TEXT_TO_IMAGE - Missing required environment variables: {', '.join(missing_vars)}")
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")

    @property
    def together_client(self) -> Together:
        """Get or create Together client instance using singleton pattern."""
        if self._together_client is None:
            if not settings.TOGETHER_API_KEY:
                raise ValueError("TOGETHER_API_KEY environment variable not set.")
            self._together_client = Together(api_key=settings.TOGETHER_API_KEY)
            logger.info("TEXT_TO_IMAGE - Together AI client initialized.")
        return self._together_client

    async def generate_image(self, prompt: str, output_path: str = "") -> bytes:
        """Generate an image from a prompt using Together AI."""
        if not prompt.strip():
            raise ValueError("Prompt cannot be empty for image generation.")

        try:
            logger.info(f"TEXT_TO_IMAGE - Generating image for prompt: '{prompt}' using model: {settings.TTI_MODEL_NAME}")

            # Together AI image generation expects the client instance, not httpx
            # Make sure Together AI client is properly awaited or is synchronous if it's blocking
            # Assuming self.together_client.images.generate is an async method if using Together SDK v1+
            response = self.together_client.images.generate(
                prompt=prompt,
                model=settings.TTI_MODEL_NAME, # TTI_MODEL_NAME is from settings
                width=1024,
                height=768,
                steps=4,
                n=1,
                response_format="b64_json",
            )
            
            # response.data is a list, response.data[0] is the object with b64_json
            if not response.data or not hasattr(response.data[0], 'b64_json'):
                raise TextToImageError("Together AI response missing image data (b64_json).")

            image_data = base64.b64decode(response.data[0].b64_json)

            if output_path:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                with open(output_path, "wb") as f:
                    f.write(image_data)
                logger.info(f"TEXT_TO_IMAGE - Image saved to {output_path}")

            return image_data

        except Exception as e:
            logger.error(f"TEXT_TO_IMAGE - Failed to generate image: {e}", exc_info=True)
            raise TextToImageError(f"Failed to generate image: {str(e)}") from e

    async def create_scenario(self, chat_history: list = None) -> ScenarioPrompt:
        """Creates a first-person narrative scenario and corresponding image prompt based on chat history."""
        if chat_history is None:
            chat_history = []
        
        try:
            # Format chat history for the prompt
            formatted_history = "\n".join([f"{msg.type.title()}: {msg.content}" for msg in chat_history[-5:]])
            logger.info("TEXT_TO_IMAGE - Creating scenario from chat history.")

            prompt_content = IMAGE_SCENARIO_PROMPT.format(chat_history=formatted_history)
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    self.groq_base_url, # Use Groq API endpoint for scenario creation
                    headers=self.groq_headers,
                    json={
                        "model": self.groq_model_name, # Use Groq model for scenario
                        "messages": [{"role": "user", "content": prompt_content}],
                        "temperature": 0.4
                    }
                )
                response.raise_for_status()
                result = response.json()
                
                content = result["choices"][0]["message"]["content"]
                # Assuming the LLM output is directly the narrative and image prompt,
                # or that you have a parsing step here if it returns JSON.
                # If it's a single string, you might just use it for both for now,
                # or guide the LLM to output specific JSON for narrative/prompt.
                logger.info(f"TEXT_TO_IMAGE - Scenario creation LLM response: {content[:100]}...")
                return ScenarioPrompt(
                    narrative=content, 
                    image_prompt=content 
                )

        except Exception as e:
            logger.error(f"TEXT_TO_IMAGE - Failed to create scenario: {e}", exc_info=True)
            raise TextToImageError(f"Failed to create scenario: {str(e)}") from e

    async def enhance_prompt(self, prompt: str) -> str:
        """Enhance a simple prompt with additional details and context."""
        if not prompt.strip():
            raise ValueError("Prompt cannot be empty for enhancement.")

        try:
            logger.info(f"TEXT_TO_IMAGE - Enhancing prompt: '{prompt}'")

            enhanced_prompt_content = IMAGE_ENHANCEMENT_PROMPT.format(prompt=prompt)
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    self.groq_base_url, # Use Groq API endpoint for prompt enhancement
                    headers=self.groq_headers,
                    json={
                        "model": self.groq_model_name,
                        "messages": [{"role": "user", "content": enhanced_prompt_content}],
                        "temperature": 0.25
                    }
                )
                response.raise_for_status()
                result = response.json()
                
                enhanced_text = result["choices"][0]["message"]["content"]
                logger.info(f"TEXT_TO_IMAGE - Prompt enhancement LLM response: {enhanced_text[:100]}...")
                return enhanced_text

        except Exception as e:
            logger.error(f"TEXT_TO_IMAGE - Failed to enhance prompt: {e}", exc_info=True)
            raise TextToImageError(f"Failed to enhance prompt: {str(e)}") from e


# --- get_text_to_image_module function (Singleton) ---
from functools import lru_cache

@lru_cache(maxsize=1)
def get_text_to_image_module() -> TextToImage:
    logger.info("TEXT_TO_IMAGE - Calling get_text_to_image_module (using lru_cache).")
    return TextToImage()