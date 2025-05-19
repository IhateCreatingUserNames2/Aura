# orchestrator_adk_agent.py
import os
import json
import asyncio
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional


from dotenv import load_dotenv
# Assuming .env is in the same directory or project root
# If orchestrator_adk_agent.py is in the root, this is fine:
dotenv_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.env')
if not os.path.exists(dotenv_path): # Fallback if it's one level up (e.g. running from a subfolder)
    dotenv_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env')

if os.path.exists(dotenv_path):
    print(f"Orchestrator ADK: Loading .env file from: {dotenv_path}")
    load_dotenv(dotenv_path)
else:
    print(f"Orchestrator ADK: .env file not found. Relying on environment variables.")
# +++++++++++++++++++++++++++++++++++++++++++++


from google.adk.agents import LlmAgent
# Import LiteLlm model wrapper from ADK
from google.adk.models.lite_llm import LiteLlm
from google.adk.tools import FunctionTool, ToolContext
from google.adk.sessions import InMemorySessionService # Or DatabaseSessionService
from google.adk.runners import Runner
from google.genai.types import Content as ADKContent, Part as ADKPart

# Import from memory_system
from memory_system.memory_blossom import MemoryBlossom
from memory_system.memory_connector import MemoryConnector
from memory_system.memory_models import Memory



# --- Configuration ---
# Make sure OPENROUTER_API_KEY is set in your environment!
# e.g., export OPENROUTER_API_KEY="sk-or-v1-..."
# LiteLLM will pick it up automatically.

# Optional OpenRouter headers for analytics/leaderboards
os.environ["OR_SITE_URL"] = "https://3e7d-189-28-2-52.ngrok-free.app" # Your site
os.environ["OR_APP_NAME"] = "Lain"    # Your app name

# --- CHOOSE YOUR OPENROUTER MODEL ---
# Option 1: Specify a model explicitly via OpenRouter
# Find model slugs on OpenRouter's website (e.g., Models page)
# AGENT_MODEL_STRING = "openrouter/google/gemini-flash-1.5"
# AGENT_MODEL_STRING = "openrouter/openai/gpt-4o-mini"
AGENT_MODEL_STRING = "openrouter/openai/gpt-4o-mini" # Example: Haiku

# Option 2: Use OpenRouter's "auto" router (NotDiamond)
# AGENT_MODEL_STRING = "openrouter/auto"

# Option 3: Use OpenRouter with web search (for supported models)
# AGENT_MODEL_STRING = "openrouter/openai/gpt-4o:online" # Note the :online suffix

AGENT_MODEL = LiteLlm(model=AGENT_MODEL_STRING) # Pass the OpenRouter model string to ADK's LiteLlm

ADK_APP_NAME = "OrchestratorMemoryApp_OpenRouter"

# --- Initialize MemoryBlossom (remains the same) ---
memory_blossom_persistence_file = os.getenv("MEMORY_BLOSSOM_PERSISTENCE_PATH", "memory_blossom_data.json")
memory_blossom_instance = MemoryBlossom(persistence_path=memory_blossom_persistence_file)
memory_connector_instance = MemoryConnector(memory_blossom_instance)
memory_blossom_instance.set_memory_connector(memory_connector_instance)

# --- ADK Tools for MemoryBlossom (remain the same) ---
def add_memory_tool_func(
    content: str,
    memory_type: str,
    emotion_score: float = 0.0,
    coherence_score: float = 0.5,
    novelty_score: float = 0.5,
    initial_salience: float = 0.5,
    metadata_json: Optional[str] = None,
    tool_context: Optional[ToolContext] = None
) -> Dict[str, Any]:
    """
    Adds a memory to the MemoryBlossom system.
    Args:
        content: The textual content of the memory.
        memory_type: The type of memory (e.g., Explicit, Emotional, Procedural, Flashbulb, Somatic, Liminal, Generative).
        emotion_score: Emotional intensity of the memory (0.0 to 1.0).
        coherence_score: How well-structured or logical the memory is (0.0 to 1.0).
        novelty_score: How unique or surprising the memory is (0.0 to 1.0).
        initial_salience: Initial importance of the memory (0.0 to 1.0).
        metadata_json: Optional JSON string representing a dictionary of additional metadata.
    """
    print(f"--- TOOL: add_memory_tool_func called with type: {memory_type} ---")
    parsed_metadata = None
    if metadata_json:
        try:
            parsed_metadata = json.loads(metadata_json)
        except json.JSONDecodeError:
            return {"status": "error", "message": "Invalid JSON format for metadata."}
    try:
        memory = memory_blossom_instance.add_memory(
            content=content,
            memory_type=memory_type,
            metadata=parsed_metadata,
            emotion_score=emotion_score,
            coherence_score=coherence_score,
            novelty_score=novelty_score,
            initial_salience=initial_salience
        )
        memory_blossom_instance.save_memories()
        return {"status": "success", "memory_id": memory.id, "message": f"Memory of type '{memory_type}' added."}
    except Exception as e:
        print(f"Error in add_memory_tool_func: {str(e)}")
        return {"status": "error", "message": str(e)}

def recall_memories_tool_func(
    query: str,
    target_memory_types_json: Optional[str] = None,
    top_k: int = 5,
    tool_context: Optional[ToolContext] = None
) -> Dict[str, Any]:
    """
    Recalls memories from the MemoryBlossom system based on a query.
    Args:
        query: The search query to find relevant memories.
        target_memory_types_json: Optional JSON string of a list of memory types to specifically search within.
        top_k: The maximum number of memories to return.
    """
    print(f"--- TOOL: recall_memories_tool_func called with query: {query[:30]}... ---")
    target_types_list: Optional[List[str]] = None
    if target_memory_types_json:
        try:
            target_types_list = json.loads(target_memory_types_json)
            if not isinstance(target_types_list, list) or not all(isinstance(item, str) for item in target_types_list):
                return {"status": "error", "message": "target_memory_types_json must be a JSON string of a list of strings."}
        except json.JSONDecodeError:
            return {"status": "error", "message": "Invalid JSON format for target_memory_types_json."}
    try:
        conversation_history = None
        if tool_context and tool_context.state:
            conversation_history = tool_context.state.get('conversation_history', [])

        recalled_memories = memory_blossom_instance.retrieve_memories(
            query=query,
            target_memory_types=target_types_list,
            top_k=top_k,
            conversation_context=conversation_history
        )
        return {
            "status": "success",
            "count": len(recalled_memories),
            "memories": [mem.to_dict() for mem in recalled_memories]
        }
    except Exception as e:
        print(f"Error in recall_memories_tool_func: {str(e)}")
        return {"status": "error", "message": str(e)}

add_memory_adk_tool = FunctionTool(func=add_memory_tool_func)
recall_memories_adk_tool = FunctionTool(func=recall_memories_tool_func)

# --- Orchestrator ADK Agent Definition (instruction largely same) ---
aura_agent_instruction = """
You are Aura, a helpful and insightful AI assistant.
The user's message you receive is a specially constructed prompt that contains rich contextual information:
- `<SYSTEM_PERSONA_START>`...`<SYSTEM_PERSONA_END>`: Defines your persona and detailed characteristics.
- `<NARRATIVE_FOUNDATION_START>`...`<NARRATIVE_FOUNDATION_END>`: Summarizes your understanding and journey with the user so far (Narrativa de Fundamento).
- `<SPECIFIC_CONTEXT_RAG_START>`...`<SPECIFIC_CONTEXT_RAG_END>`: Provides specific information retrieved (RAG) relevant to the user's current query.
- `<RECENT_HISTORY_START>`...`<RECENT_HISTORY_END>`: Shows the recent turns of your conversation.
- `<CURRENT_SITUATION_START>`...`<CURRENT_SITUATION_END>`: Includes the user's latest raw reply and your primary task.

Your main goal is to synthesize ALL this provided information to generate a comprehensive, coherent, and natural response to the user's latest reply indicated in the "Situação Atual" section.
Actively acknowledge and weave in elements from the "Narrativa de Fundamento" and "Informações RAG" into your response to show deep understanding and context.
Maintain the persona defined.

If, after considering all the provided context, you believe new information from the current interaction needs to be stored for long-term recall, or if you need to perform a very specific targeted memory search not covered by the RAG, you may use the following tools:
- 'add_memory_tool_func': To store new important memories.
  - You MUST specify 'memory_type'. Choose from: Explicit, Emotional, Procedural, Flashbulb, Somatic, Liminal, Generative.
  - Briefly explain your choice of memory_type to the user.
- 'recall_memories_tool_func': To search for specific memories if the provided RAG is insufficient and you have a clear query.

Strive for insightful, helpful, and contextually rich interactions.
If you identify a potential contradiction between the provided context pieces (e.g., RAG vs. Foundation Narrative), try to address it gracefully in your response, perhaps by prioritizing the most recent or specific information, or by noting the differing perspectives.
"""

orchestrator_adk_agent_aura = LlmAgent(
    name="AuraNCFOrchestratorOpenRouter",
    model=AGENT_MODEL,
    instruction=aura_agent_instruction,
    tools=[add_memory_adk_tool, recall_memories_adk_tool],
)

# ADK Runner and Session Service (sem mudanças aqui)
adk_session_service = InMemorySessionService()
adk_runner = Runner(
    agent=orchestrator_adk_agent_aura,
    app_name=ADK_APP_NAME,
    session_service=adk_session_service
)

# --- Helper for Aura-Reflector to add memories programmatically ---
# This bypasses the need for the main LLM to use a tool if the Reflector decides to store something.
def reflector_add_memory(
    content: str,
    memory_type: str,
    emotion_score: float = 0.0,
    coherence_score: float = 0.5,
    novelty_score: float = 0.5,
    initial_salience: float = 0.5,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Allows the Aura-Reflector to add a memory directly."""
    print(f"--- REFLECTOR: Adding memory of type: {memory_type} ---")
    try:
        memory = memory_blossom_instance.add_memory(
            content=content,
            memory_type=memory_type,
            metadata=metadata, # metadata is already a dict here
            emotion_score=emotion_score,
            coherence_score=coherence_score,
            novelty_score=novelty_score,
            initial_salience=initial_salience
        )
        memory_blossom_instance.save_memories()
        return {"status": "success", "memory_id": memory.id, "message": f"Reflector added memory of type '{memory_type}'."}
    except Exception as e:
        print(f"Error in reflector_add_memory: {str(e)}")
        return {"status": "error", "message": str(e)}

print(f"ADK Aura Agent '{orchestrator_adk_agent_aura.name}' and Runner initialized with LiteLLM model '{AGENT_MODEL_STRING}'.")
print(f"MemoryBlossom instance ready. Loaded {sum(len(m) for m in memory_blossom_instance.memory_stores.values())} memories from persistence.")

# --- Standalone Test Function (remains same, but will use OpenRouter now) ---
async def run_adk_test_conversation():
    user_id = "test_user_adk_openrouter"
    session_id = "test_session_adk_openrouter"

    _ = adk_session_service.get_session(app_name=ADK_APP_NAME, user_id=user_id, session_id=session_id) or \
        adk_session_service.create_session(app_name=ADK_APP_NAME, user_id=user_id, session_id=session_id, state={'conversation_history': []})

    queries = [
        "Hello! I'm exploring how AI can manage different types of memories.",
        "Please remember that my research focus is 'emergent narrative structures' and I find 'liminal spaces' fascinating. Store this as an Explicit memory, with high novelty.",
        "What is my research focus and what do I find fascinating?",
        "Let's try storing an emotional memory: 'The sunset over the mountains was breathtakingly beautiful, a truly awe-inspiring moment.' Set emotion_score to 0.9.",
        "What beautiful moment did I describe?",
        "Can you search for memories related to 'sunsets' or 'mountains' in my Emotional memories? (target_memory_types_json='[\"Emotional\"]')",
        "Thank you, that's all for now."
    ]
    current_adk_session_state = adk_runner.session_service.get_session(ADK_APP_NAME, user_id, session_id).state
    if 'conversation_history' not in current_adk_session_state:
        current_adk_session_state['conversation_history'] = []

    for query in queries:
        print(f"\nUSER: {query}")
        current_adk_session_state['conversation_history'].append({"role": "user", "content": query})
        adk_input_content = ADKContent(role="user", parts=[ADKPart(text=query)])
        final_response_text = "ADK Agent: (No final text response)"
        async for event in adk_runner.run_async(
            user_id=user_id, session_id=session_id, new_message=adk_input_content
        ):
            if event.is_final_response():
                if event.content and event.content.parts and event.content.parts[0].text:
                    final_response_text = event.content.parts[0].text.strip()
                break
        print(f"AGENT: {final_response_text}")
        current_adk_session_state['conversation_history'].append({"role": "assistant", "content": final_response_text})
        current_adk_session_state = adk_runner.session_service.get_session(ADK_APP_NAME, user_id, session_id).state


if __name__ == "__main__":
    if not os.getenv("OPENROUTER_API_KEY"):
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!! ERROR: OPENROUTER_API_KEY environment variable is not set.             !!!")
        print("!!! The agent will likely fail to make LLM calls.                          !!!")
        print("!!! Please set it before running. e.g., export OPENROUTER_API_KEY='sk-or-...' !!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print("Aura ADK Agent (using OpenRouter via LiteLLM NCF Prompt) and MemoryBlossom are set up.")
    print("To test, use the A2A wrapper.")