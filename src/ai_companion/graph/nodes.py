import datetime
from math import e
import os
from uuid import uuid4
import re
import chainlit as cl
import spacy

from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage
from langchain_core.runnables import RunnableConfig
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.messages import SystemMessage
from typing import Dict, Any

from ai_companion.modules.image.image_tools_call import generate_ava_image
from ai_companion.core.prompts import CHARACTER_CARD_PROMPT, ROUTER_PROMPT
from ai_companion.graph.utils.chains import (
    get_character_response_chain,
    get_router_chain,
)
from ai_companion.graph.utils.helpers import (
    get_chat_model,
    get_text_to_image_module,
    get_text_to_speech_module,
    AsteriskRemovalParser,
)
from ai_companion.modules.memory.long_term.memory_manager import get_memory_manager
from ai_companion.modules.schedules.context_generation import ScheduleContextGenerator
from ai_companion.settings import settings

# Load the spaCy English language model (do this once, maybe globally)
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    cl.logger.warning("Downloading spaCy en_core_web_sm model...")
    spacy.cli.download("en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")

async def router_node(state: Dict[str, Any], config: RunnableConfig):
    """Route the conversation to the appropriate workflow"""
    messages = state.get("messages", [])
    cl.logger.info(f"ROUTER_NODE - config: {config}")
    cl.logger.info(f"ROUTER_NODE - state: {state}")
    if not messages:
        return {"workflow": "conversation", "messages": [], "user_name": state.get("current_state", {}).get("user_name", config.get("configurable", {}).get("user_name", "")), "thread_id": state.get("thread_id", config.get("configurable", {}).get("thread_id", ""))}

    chain = await get_router_chain()
    response = await chain.ainvoke({"messages": messages[-settings.ROUTER_MESSAGES_TO_ANALYZE :]})
    cl.logger.info(f"ROUTER_NODE - Routing decision: {response.response_type}")
    # For initial messages, default to conversation
    if len(messages) == 1 and response.response_type == "conversation":
        cl.logger.info(f"ROUTER_NODE - Initial message detected, defaulting to conversation workflow. With user_name: {state.get('current_state', {}).get('user_name', config.get('configurable', {}).get('user_name', ''))} and thread_id: {state.get('thread_id', config.get('configurable', {}).get('thread_id', ''))}")
        return {"workflow": "conversation", "messages": messages, "user_name": state.get("current_state", {}).get("user_name", config.get("configurable", {}).get("user_name", "")), "thread_id": state.get("thread_id", config.get("configurable", {}).get("thread_id", ""))}

    return {"workflow": response.response_type, "messages": messages, "user_name": state.get("current_state", {}).get("user_name", config.get("configurable", {}).get("user_name", "")), "thread_id": state.get("thread_id", config.get("configurable", {}).get("thread_id", ""))}


def context_injection_node(state: Dict[str, Any], config: RunnableConfig):
    schedule_context = ScheduleContextGenerator.get_current_activity()
    #cl.logger.info(f"NODES - context_injection_node - state: {state}")
    #cl.logger.info(f"NODES - context_injection_node - config: {config}")
    cl.logger.info(f"NODES - context_injection_node - user_name: {state['user_name']}")
    if schedule_context != state.get("current_activity", ""):
        apply_activity = True
    else:
        apply_activity = False
    return {
        "apply_activity": apply_activity, 
        "current_activity": schedule_context,
        "messages": state.get("messages", []),
        "current_state": state,
        "user_name": state.get("user_name", ""),
        "thread_id": state.get("thread_id", "")
    }

async def conversation_node(state: Dict[str, Any], config: RunnableConfig):
    """Generate a response using the character's personality and conversation history."""
    messages = state.get("messages", [])
    #cl.logger.info(f"NODES - conversation_node - state: {state}")
    cl.logger.info(f"NODES - conversation_node - config: {config}")
    if not messages:
        return {"messages": [AIMessage(content="I'm sorry, I didn't receive any messages to respond to.")]}

    try:
        memory_context = state.get("memory_context", "")
        current_activity = state.get("current_activity", "")
        user_name = state.get("current_state", {}).get("user_name", config.get("configurable", {}).get("user_name", ""))
        cl.logger.info(f"NODES - conversation_node - memory_context: {memory_context}, current_activity: {current_activity}, user_name: {user_name}")
        thread_id = config.get("configurable", {}).get("thread_id")
        cl.logger.info(f"NODES - conversation_node - thread_id from config: {thread_id}, user_name from config: {user_name}")

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

        prompt_kwargs = {
            "messages": messages,
            "user_name": user_name,
            "memory_context": memory_context,
            "current_activity": current_activity,
        }

        model = await get_chat_model()
        llm_with_tools = model.bind_tools([generate_ava_image])
        chain = prompt | llm_with_tools | AsteriskRemovalParser()

        response = await chain.ainvoke(prompt_kwargs)

        if isinstance(response, str):
            return {"messages": [*messages, AIMessage(content=response)]}  # Ensure AIMessage
        elif isinstance(response, list):
            return {"messages": [*messages, *response]}  # Ensure list of AIMessages
        else:
            return {"messages": [*messages, AIMessage(content=str(response))]}  # Ensure AIMessage
    except Exception as e:
        # --- CRITICAL FIX: Re-raise the exception ---
        cl.logger.error(f"Error in conversation_node: {e}", exc_info=True) # Log it
        raise # Re-raise the exception to be caught by app.py's try-except
        # --- END CRITICAL FIX ---

async def image_node(state: Dict[str, Any], config: RunnableConfig):
    current_activity = ScheduleContextGenerator.get_current_activity()
    cl.logger.info(f"NODES - image_node - current_activity: {current_activity}")
    cl.logger.info(f"NODES - image_node - state: {state}")
    cl.logger.info(f"NODES - image_node - messages: {state.get('messages', [])}")
    cl.logger.info(f"NODES - image_node - config: {config}")
    memory_context = state.get("memory_context", "")

    user_name = state.get("current_state", {}).get("user_name", config.get("configurable", {}).get("user_name", ""))
    cl.logger.info(f"NODES - image_node - memory_context: {memory_context}, current_activity: {current_activity}, user_name: {user_name}")
    thread_id = config.get("configurable", {}).get("thread_id")
    cl.logger.info(f"NODES - image_node - thread_id from config: {thread_id}, user_name from config: {user_name}")

    chain = await get_character_response_chain(state, config)
    cl.logger.info(f"NODES - image_node - chain obtained: {chain}")
    text_to_image_module = get_text_to_image_module()

    scenario = await text_to_image_module.create_scenario(state.get("messages", [])[-5:])
    os.makedirs("generated_images", exist_ok=True)
    img_path = f"generated_images/image_{str(uuid4())}.png"
    await text_to_image_module.generate_image(scenario.image_prompt, img_path)

    # Inject the image prompt information as an AI message
    scenario_message = HumanMessage(content=f"<image attached by Ava generated from prompt: {scenario.image_prompt}>")
    updated_messages = state.get("messages", []) + [scenario_message]

    response = await chain.ainvoke(
        {
            "messages": updated_messages,
            "current_activity": current_activity,
            "memory_context": memory_context,
            "user_name": user_name, # Pass user_name
            # If your get_character_response_chain also uses thread_id, pass it too
            "thread_id": thread_id
        }
    )

    # Ensure return type is consistent
    if isinstance(response, str):
        response_message = AIMessage(content=response)
    elif isinstance(response, list):
            return {"messages": [*updated_messages, *response]}
    else: # Fallback
        response_message = AIMessage(content=str(response))

    return {
        "messages": state.get("messages", []) + [response_message], # Append to original messages for state propagation
        "image_path": img_path,
        "workflow": "image", # Explicitly set workflow if image_node dictates it
        "user_name": user_name, # Propagate user_name
        "current_activity": current_activity, # Propagate
        "memory_context": memory_context, # Propagate
        "thread_id": thread_id # Propagate
    }


async def audio_node(state: Dict[str, Any], config: RunnableConfig):
    current_activity = ScheduleContextGenerator.get_current_activity()
    cl.logger.info(f"NODES - audio_node - current_activity: {current_activity}")
    cl.logger.info(f"NODES - audio_node - state: {state}")
    cl.logger.info(f"NODES - audio_node - messages: {state.get('messages', [])}")
    cl.logger.info(f"NODES - audio_node - config: {config}")
    memory_context = state.get("memory_context", "")

    user_name = state.get("current_state", {}).get("user_name", config.get("configurable", {}).get("user_name", ""))
    cl.logger.info(f"NODES - audio_node - memory_context: {memory_context}, current_activity: {current_activity}, user_name: {user_name}")
    thread_id = config.get("configurable", {}).get("thread_id")
    cl.logger.info(f"NODES - audio_node - thread_id from config: {thread_id}, user_name from config: {user_name}")

    chain = await get_character_response_chain(state, config)
    cl.logger.info(f"NODES - audio_node - chain obtained: {chain}")
    text_to_speech_module = get_text_to_speech_module()

    response = await chain.ainvoke(
        {
            "messages": state.get("messages", []),
            "current_activity": current_activity,
            "memory_context": memory_context,
        },
        config,
    )
    cl.logger.info(f"NODES - audio_node - response from AI chain: {response}")
    output_audio = await text_to_speech_module.synthesize(response)
    cl.logger.info(f"NODES - audio_node - output_audio length : {len(output_audio)} bytes")

    return {
        "messages": state.get("messages", []) + [AIMessage(content=response)] + [output_audio], # Append to original messages for state propagation
        "audio_buffer": output_audio,
        "workflow": "audio", # Explicitly set workflow if audio_node dictates it
        "user_name": user_name, # Propagate user_name
        "current_activity": current_activity, # Propagate
        "memory_context": memory_context, # Propagate
        "thread_id": thread_id # Propagate
    }


async def summarize_conversation_node(state: Dict[str, Any], config: RunnableConfig):
    model = await get_chat_model()
    summary = state.get("summary", "")
    cl.logger.info(f"NODES - summarize_conversation_node - config: {config}")
    cl.logger.info(f"NODES - summarize_conversation_node - state: {state}")

    if summary:
        summary_message = (
            f"This is summary of the conversation to date between Ava and the user: {summary}\n\n"
            "Extend the summary by taking into account the new messages above:"
        )
    else:
        summary_message = (
            "Create a summary of the conversation above between Ava and the user. "
            "The summary must be a short description of the conversation so far, "
            "but that captures all the relevant information shared between Ava and the user:"
        )

    messages = state.get("messages", []) + [HumanMessage(content=summary_message)]
    response = await model.ainvoke(messages)

    delete_messages = [RemoveMessage(id=m.id) for m in state.get("messages", [])[: -settings.TOTAL_MESSAGES_AFTER_SUMMARY]]
    return {"summary": response.content, "messages": delete_messages, "current_state": state, "user_name": state.get("user_name", config.get("configurable", {}).get("user_name", "")), "thread_id": state.get("thread_id", config.get("configurable", {}).get("thread_id", ""))}

async def memory_extraction_node(state: Dict[str, Any], config: RunnableConfig):
    """Extract and store the user's name if provided, or if the chatbot just asked for it."""
    messages = state.get("messages", [])
    if not messages:  # Need at least two messages to check history
        cl.logger.info("MEMORY_EXTRACTION_NODE - Not enough messages for name check.")
        return state

    last_user_message = None
    previous_bot_message = None

    # Iterate through the list to find the last human and the immediately preceding AI message
    for i in range(len(messages) - 1, -1, -1):
        message = messages[i]
        if isinstance(message, HumanMessage) and last_user_message is None:
            last_user_message = message
            break

    # If we found a last user message, look for the most recent bot message before it
    if last_user_message:
        cl.logger.info(f"MEMORY_EXTRACTION_NODE - Found last user message: {last_user_message.content} at index {messages.index(last_user_message)}")
        if len(messages) > 1 and messages.index(last_user_message) >= 0:
            cl.logger.info(f"MEMORY_EXTRACTION_NODE - Looking for previous bot message after index {messages.index(last_user_message)}")
            if isinstance(messages[1], AIMessage):
                previous_bot_message = messages[1]
                cl.logger.info(f"MEMORY_EXTRACTION_NODE - Found previous bot message: {previous_bot_message.content} for index {1}")
        else:
            cl.logger.info("MEMORY_EXTRACTION_NODE - No previous bot message found.")
            previous_bot_message = messages[1] 
        
    memory_manager = get_memory_manager()
    extracted_name = None # Declare extracted_name at the top

    if last_user_message and previous_bot_message:
        cl.logger.info(f"MEMORY_EXTRACTION_NODE - Last user message: '{last_user_message.content}'")
        cl.logger.info(f"MEMORY_EXTRACTION_NODE - Previous bot message: '{previous_bot_message.content}'")

        if "your name" in previous_bot_message.content.lower():
            cl.logger.info("MEMORY_EXTRACTION_NODE - Bot asked for user's name.")
            name_match = re.search(r"(?:i'm|i am|my name is)\s+(\w+)|(?:call me|you can call me)\s+(\w+)", last_user_message.content, re.IGNORECASE)
            cl.logger.info(f"MEMORY_EXTRACTION_NODE - Name match: {name_match}")
            if name_match:
                extracted_name = name_match.group(1) or name_match.group(2)
                cl.logger.info(f"MEMORY_EXTRACTION_NODE - Extracted name in response to bot's question: {extracted_name}")
                await memory_manager.vector_store.store_memory(
                    text=f"Is named {extracted_name}",
                    metadata={"id": str(uuid4()), "timestamp": datetime.datetime.now().replace(microsecond=0).isoformat(), "type": "user_name"}
                )
                #return {"messages": messages}
        elif previous_bot_message.content.lower().startswith("hi") or previous_bot_message.content.lower().startswith("hey"):
            cl.logger.info("MEMORY_EXTRACTION_NODE - Bot got user's name.")
            name_match = re.search(
                r"(?:i'm|i am|my name is)\s+([\w\s]+)(?:,|\.)?|"
                r"(?:call me|you can call me)\s+([\w\s]+)(?:,|\.)?|"
                r"(?:this is|myself)\s+([\w\s]+)(?:,|\.)?|"
                r"([\w\s]+)\s+here(?:,|\.)?",
                last_user_message.content,
                re.IGNORECASE
            )
            cl.logger.info(f"MEMORY_EXTRACTION_NODE - Name match: {name_match}")
            if name_match:
                extracted_name = (name_match.group(1) or name_match.group(2) or name_match.group(3) or name_match.group(4)).strip()
                cl.logger.info(f"MEMORY_EXTRACTION_NODE - Extracted name in response to bot's question: {extracted_name}")
                await memory_manager.vector_store.store_memory(
                    text=f"Is named {extracted_name}",
                    metadata={"id": str(uuid4()), "timestamp": datetime.datetime.now().replace(microsecond=0).isoformat(), "type": "user_name"}
                )
                #return {"messages": messages}        

    # Extract other relevant memories from the last user message
    if last_user_message:
        analysis = await memory_manager._analyze_memory(last_user_message.content)
        if analysis.is_important and analysis.formatted_memory is not None and analysis.formatted_memory.startswith("Is named "):
            similar = memory_manager.vector_store.find_similar_memory(analysis.formatted_memory)
            if not similar:
                metadata = {"id": str(uuid4()), "timestamp": datetime.datetime.now().replace(microsecond=0).isoformat(), "type": "user_name"}
                await memory_manager.vector_store.store_memory(text=analysis.formatted_memory, metadata=metadata)
                cl.logger.info(f"MEMORY_EXTRACTION_NODE - Stored general memory: '{analysis.formatted_memory}'")
            elif analysis.is_important and analysis.formatted_memory is None:
                cl.logger.info("MEMORY_EXTRACTION_NODE - formatted_memory is None, skipping general memory storage.")   

    if extracted_name:
        config["configurable"]["user_name"] = extracted_name
        state["user_name"] = extracted_name
        cl.logger.info(f"MEMORY_EXTRACTION_NODE - Updated user_name in config: {config['configurable']['user_name']} and state: {state['user_name']}")

    return {"messages": messages,"current_state": state, "user_name": extracted_name}  # Return the updated config

async def memory_injection_node(state: Dict[str, Any], config: RunnableConfig):
    """Retrieve and inject relevant memories, including the user's name."""
    memory_manager = get_memory_manager()
    current_user_name = state.get("current_state", {}).get("user_name", config.get("configurable", {}).get("user_name", ""))
    thread_id = config.get("configurable", {}).get("thread_id") # Log thread_id as well
    cl.logger.info(f"NODES - memory_injection_node - thread_id from config: {thread_id}, user_name from config: {current_user_name}")
    # --- CRITICAL CHANGE: Explicitly search for the user's name first ---
    actual_user_name_from_memory = None
    current_activity = ScheduleContextGenerator.get_current_activity()

    try:
        # Search for memories explicitly marked as 'user_name' type or 'Is named X'
        # This is a direct search to confirm if the name has been stored.
        # You might have a specific vector store query for 'type: user_name' if your
        # vector store supports filtering by metadata during search.
        # For now, let's just search for common phrases that would recall a name.
        
        # Query based on the current user_name in config (which might be thread_id or actual name)
        # Or, ideally, a very specific query that recalls the 'type: user_name' entry.
        
        # Option A: Direct lookup (if your vector store allows getting by specific metadata key)
        # If your vector store had a way to retrieve a specific memory by a unique key/type, use it here.
        # e.g., result = await memory_manager.vector_store.get_memory_by_type("user_name")

        # Option B: Semantic search for "my name is" context
        name_search_query = "What is the user's name?" # A generic query to find names
        name_search_results = await memory_manager.vector_store.search_memories(name_search_query, k=1)
        
        if name_search_results:
            # Check the metadata of the found memory for 'user_name' type
            for result_mem in name_search_results:
                if result_mem.metadata.get("type") == "user_name" and "user_name" in result_mem.metadata:
                    actual_user_name_from_memory = result_mem.metadata["user_name"]
                    cl.logger.info(f"MEMORY_INJECTION_NODE - Found actual user_name from memory search: {actual_user_name_from_memory}")
                    break
                elif result_mem.text.startswith("Is named "):
                    actual_user_name_from_memory = result_mem.text.replace("Is named ", "", 1).strip()
                    cl.logger.info(f"MEMORY_INJECTION_NODE - Found actual user_name from memory text: {actual_user_name_from_memory}")
                    break

    except Exception as e:
        cl.logger.error(f"MEMORY_INJECTION_NODE - Error searching for user name in memory: {e}", exc_info=True)
        actual_user_name_from_memory = None

    # Prioritize the extracted name over the config's initial thread_id
    current_user_name = actual_user_name_from_memory if actual_user_name_from_memory else current_user_name
    # --- END CRITICAL CHANGE ---
    # Get general relevant memories (using a more focused context if needed)
    # The 'recent_context' for general memories might still be the full recent conversation.
    messages = state.get("messages", [])
    cl.logger.info(f"NODES - memory_injection_node - messages length: {len(messages)}, messages: {messages[-3:] if len(messages) >= 3 else messages}")
    recent_context_for_general_memories = " ".join([m.content for m in messages[-3:] if isinstance(m, (HumanMessage, AIMessage))]) # Still full recent context
    relevant_memories = await memory_manager.vector_store.search_memories(recent_context_for_general_memories, k=settings.MEMORY_TOP_K)
    
    cl.logger.info(f"MEMORY_INJECTION_NODE - Relevant general memories: {[mem.text for mem in relevant_memories]}")
    memory_context = memory_manager.format_memories_for_prompt([mem.text for mem in relevant_memories])

    cl.logger.info(f"NODES - memory_injection_node - injecting user_name: {current_user_name}")
    return {  # Whether to apply the current activity
        "memory_context": memory_context,
        "current_state": state,  # Inject the potentially updated user_name
        "messages": messages,
        "user_name": current_user_name,  # Inject the user's name
        "current_activity": current_activity,  # Inject the current activity
        "thread_id": thread_id  # Inject the thread_id
    }

