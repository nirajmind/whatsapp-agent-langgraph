from functools import lru_cache

from langgraph.graph import END, START, StateGraph
from ai_companion.modules.image.image_tools_call import generate_ava_image
from langgraph.prebuilt import ToolNode

from ai_companion.graph.edges import (
    select_workflow,
    should_summarize_conversation,
)
from ai_companion.graph.nodes import (
    audio_node,
    context_injection_node,
    conversation_node,
    image_node,
    memory_extraction_node,
    memory_injection_node,
    router_node,
    summarize_conversation_node,
)


@lru_cache(maxsize=1)
def create_workflow_graph():
    graph_builder = StateGraph(dict)

    # Add all nodes
    graph_builder.add_node("memory_extraction_node", memory_extraction_node)
    graph_builder.add_node("router_node", router_node)
    graph_builder.add_node("context_injection_node", context_injection_node)
    graph_builder.add_node("memory_injection_node", memory_injection_node)
    graph_builder.add_node("conversation_node", conversation_node)
    graph_builder.add_node("image_node", image_node)
    graph_builder.add_node("audio_node", audio_node)
    graph_builder.add_node("summarize_conversation_node", summarize_conversation_node)

    # --- CRITICAL FIX: Add ToolNode ---
    tools = [generate_ava_image] # List of all tools your agent can use
    tool_node = ToolNode(tools)
    graph_builder.add_node("tool_node", tool_node)
    # --- END CRITICAL FIX ---

    # Define the flow
    # First determine response type
    graph_builder.add_edge(START, "router_node")

    # Conditional edge from router to different workflows
    graph_builder.add_conditional_edges("router_node", select_workflow)

    graph_builder.add_conditional_edges(
        "conversation_node",
        # Custom routing function: if last message has tool_calls, go to tool_node
        # else continue to memory extraction
        lambda state: "tool_node" if state.get("messages", [])[-1].tool_calls else "memory_extraction_node"
    )
    # After tool execution, go to memory extraction (or directly to context injection)
    graph_builder.add_edge("tool_node", "memory_extraction_node")
    # Conversation workflow flow
    graph_builder.add_edge("memory_extraction_node", "context_injection_node")
    graph_builder.add_edge("context_injection_node", "memory_injection_node")
    graph_builder.add_conditional_edges("memory_injection_node", should_summarize_conversation)

    # Image workflow flow
    #graph_builder.add_edge("image_node", "memory_extraction_node") # Consider if memory extraction is needed here
    #graph_builder.add_conditional_edges("image_node", should_summarize_conversation)

    # Audio workflow flow
    graph_builder.add_edge("audio_node", "memory_extraction_node") # Consider if memory extraction is needed here
    graph_builder.add_conditional_edges("audio_node", should_summarize_conversation)

    # Summarization at the end
    graph_builder.add_edge("summarize_conversation_node", END)

    return graph_builder


# Compiled without a checkpointer. Used for LangGraph Studio
graph = create_workflow_graph().compile()

# --- New: Function to print graph as Mermaid syntax ---
def print_graph_mermaid():
    # Calling .get_graph() on the compiled graph returns a Graph object
    # Calling .draw_mermaid() on that Graph object gives the Mermaid string
    mermaid_string = graph.get_graph().draw_mermaid()
    print("\n--- LangGraph Mermaid Diagram ---\n")
    print("```mermaid") # Markdown fence for Mermaid
    print(mermaid_string)
    print("```")
    print("\n--- End LangGraph Mermaid Diagram ---\n")

print_graph_mermaid()    
