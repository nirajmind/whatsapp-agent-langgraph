from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.messages import SystemMessage
from typing import Dict, Any
from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser

from ai_companion.core.prompts import CHARACTER_CARD_PROMPT, ROUTER_PROMPT
from ai_companion.graph.utils.helpers import AsteriskRemovalParser, get_chat_model


class RouterResponse(BaseModel):
    response_type: str = Field(
        description="The response type to give to the user. It must be one of: 'conversation', 'image' or 'audio'"
    )


async def get_router_chain():
    model = await get_chat_model(temperature=0.3)
    parser = PydanticOutputParser(pydantic_object=RouterResponse)
    
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", ROUTER_PROMPT),
            MessagesPlaceholder(variable_name="messages")
        ]
    )

    # Add default response type for initial messages
    prompt = prompt.partial(
        default_response_type="conversation"
    )

    chain = prompt | model | parser
    return chain


async def get_character_response_chain(state: Dict[str, Any], config: RunnableConfig):
    model = await get_chat_model()
    user_name = state.get("current_state", {}).get("user_name", config.get("configurable", {}).get("user_name", ""))
    memory_context = state.get("memory_context", "")
    current_activity = state.get("current_activity", "")
    messages = state.get("messages", [])
    system_prompt_template = PromptTemplate(
            template=CHARACTER_CARD_PROMPT,
            input_variables=["user_name", "memory_context", "current_activity"],
            template_format="jinja2"  # Explicitly set the template format
        )
    system_message = SystemMessage(
        content=system_prompt_template.format(
            user_name=user_name if user_name else "",
            memory_context=memory_context,
            current_activity=current_activity
        )
    )

    prompt = ChatPromptTemplate.from_messages([
        system_message,
        MessagesPlaceholder(variable_name="messages")
    ])

    return prompt | model | AsteriskRemovalParser()
