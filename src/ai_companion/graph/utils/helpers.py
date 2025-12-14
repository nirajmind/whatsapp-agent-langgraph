from calendar import c
import re
import httpx
import chainlit as cl

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.outputs import ChatGeneration, ChatResult

from ai_companion.modules.image.image_to_text import ImageToText
from ai_companion.modules.image.text_to_image import TextToImage
from ai_companion.modules.speech import TextToSpeech
from ai_companion.settings import settings
from langchain_groq import ChatGroq 
from langchain_openai import ChatOpenAI

from pydantic import BaseModel, Field
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class XAIChatModel(BaseChatModel, BaseModel):
    temperature: float = Field(default=0.7, description="The temperature to use for generation")

    def __init__(self, temperature: float = 0.7):
        super().__init__()
        self.temperature = temperature

    @property
    def _llm_type(self) -> str:
        return "xai-chat"

    def _generate(self, messages, **kwargs):
        raise NotImplementedError("Synchronous generation not supported")

    async def _agenerate(self, messages, **kwargs):
        base_url = "https://api.x.ai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {settings.GROQ_API_KEY}",
            "Content-Type": "application/json"
        }
        
        timeout = httpx.Timeout(30.0, connect=10.0)  # 30 seconds total, 10 seconds for connection
        
        # Map message types to x.ai roles
        role_map = {
            "human": "user",
            "ai": "assistant",
            "system": "system"
        }
        
        async with httpx.AsyncClient(timeout=timeout) as client:
            try:
                response = await client.post(
                    base_url,
                    headers=headers,
                    json={
                        "model": settings.TEXT_MODEL_NAME,
                        "temperature": kwargs.get("temperature", self.temperature),
                        "messages": [{"role": role_map.get(msg.type, "user"), "content": msg.content} for msg in messages]
                    }
                )
                response.raise_for_status()
                result = response.json()
                message = AIMessage(content=result["choices"][0]["message"]["content"])
                generation = ChatGeneration(message=message)
                return ChatResult(generations=[generation])
            except httpx.TimeoutException:
                raise Exception("Request timed out. Please try again.")
            except httpx.HTTPStatusError as e:
                raise Exception(f"HTTP error occurred: {e.response.status_code} - {e.response.text}")
            except Exception as e:
                raise Exception(f"An error occurred: {str(e)}")


#async def get_chat_model(temperature: float = 0.7):
#    return XAIChatModel(temperature=temperature)
async def get_chat_model(temperature: float = 0.7) -> BaseChatModel:
    """
    Returns an instance of ChatOpenAI configured for tool calling.
    """
    # Ensure TEXT_MODEL_NAME is a OpenAI model that supports tools (e.g., "grok-3-latest", "llama3-8b-8192")
    cl.logger.info(f"Initializing ChatOpenAI with model: {settings.TEXT_MODEL_NAME} and temperature: {temperature}")
    model_name = settings.ITT_MODEL_NAME
    openai_api_key = settings.OPENAI_API_KEY

    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set.")

    # Instantiate ChatOpenAI
    try:
        chat_model = ChatOpenAI(
            temperature=temperature,
            model_name=model_name,
            api_key=openai_api_key
        )
        return chat_model
    except Exception as e:
        logger.error(f"Error in helper to get the chat model: {e}", exc_info=True)
        raise Exception(f"Failed to initialize ChatOpenAI: {str(e)}")


def get_text_to_speech_module():
    return TextToSpeech()


def get_text_to_image_module():
    return TextToImage()


def get_image_to_text_module():
    return ImageToText()


def remove_asterisk_content(text: str) -> str:
    """Remove content between asterisks from the text."""
    return re.sub(r"\*.*?\*", "", text).strip()


class AsteriskRemovalParser(StrOutputParser):
    def parse(self, text):
        return remove_asterisk_content(super().parse(text))
