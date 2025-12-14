# ai_companion/graph/tools.py
import chainlit as cl
from langchain_core.tools import tool
import os
from uuid import uuid4
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import your TextToImage module
from ai_companion.graph.utils.helpers import (
    get_text_to_image_module,
    AsteriskRemovalParser,
)

@tool
async def generate_ava_image(user_request: str) -> str:
    """
    Generates a picture of Ava based on the user's request.
    This function should be called when the user explicitly asks for an image of Ava.
    Input should describe what kind of picture the user wants.
    Example: "a picture of you chilling at a cafe", "a pic of Ava on a hike"
    """
    cl.logger.info(f"generate_ava_image called with user_request: {user_request}")

    try:
        text_to_image_module = get_text_to_image_module()
        # Use the user_request to create a scenario and then an image prompt
        # You might need to refine create_scenario or add a new method in TextToImage
        # that takes a direct user request and converts it to an image prompt.
        
        # For simplicity, let's assume create_scenario or enhance_prompt can work with user_request.
        # Or, a simpler direct prompt for now:
        # enhanced_image_prompt = await text_to_image_module.enhance_prompt(user_request)
        # scenario = await text_to_image_module.create_scenario(user_request) # If it can take string
        
        # Alternative: Just use the user_request as the prompt directly
        image_prompt_for_model = user_request # Simplest for initial testing

        os.makedirs("generated_images", exist_ok=True)
        img_path = f"generated_images/image_{str(uuid4())}.png"
        
        # generate_image returns bytes, but tool needs path/URL. Save it and return path.
        _ = await text_to_image_module.generate_image(image_prompt_for_model, img_path)

        return img_path # Return the local path to the generated image
    except Exception as e:
        # Log the error, but return a string indicating failure to the LLM
        return f"Failed to generate image: {e}"