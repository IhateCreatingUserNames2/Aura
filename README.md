# Aura: Advanced Conversational AI with Narrative Context Framing and A2A Interoperability

Aura is getting a integration with AppAgentX : https://github.com/Westlake-AGI-Lab/AppAgentX 
src https://github.com/IhateCreatingUserNames2/Aura_AppAgentX 

Aura is a prototype conversational AI agent designed for deep, contextual understanding over extended interactions. It leverages a unique **Narrative Context Framing (NCF)** prompting strategy, a sophisticated long-term memory system called **MemoryBlossom**, and an autonomous **Reflector** module for intelligent memory consolidation. Aura is exposed via the Agent-to-Agent (A2A) protocol for interoperability.

This project demonstrates advanced concepts in agent architecture, including multi-layered memory, dynamic prompt construction, and autonomous learning from interactions.



DEMO: https://aura-dahu.onrender.com/  (if its offline it is because its running on my PC and i turned it off) 

Aura Thru AiraHub In Claude Desktop
![image](https://github.com/user-attachments/assets/31dd6a52-aa1f-4018-9903-41369331eb7a)

Aira hub CURL to register Aura on Airahub: 

      curl -X POST -H "Content-Type: application/json" -d "{\"url\":\"https://b0db-189-28-2-171.ngrok-free.app\",\"name\":\"Aura2_NCF_A2A_Unified\",\"description\":\"A conversational AI agent, Aura2, with advanced memory (NCF). Exposed via A2A and ngrok.\",\"version\":\"1.2.1-unified\",\"mcp_tools\":[{\"name\":\"Aura2_NCF_narrative_conversation\",\"description\":\"Engage in a deep, contextual conversation. Aura2 uses its MemoryBlossom system and Narrative Context Framing to understand and build upon previous interactions.\",\"inputSchema\":{\"type\":\"object\",\"properties\":{\"user_input\":{\"type\":\"string\",\"description\":\"The textual input from the user for the conversation.\"},\"a2a_task_id_override\":{\"type\":\"string\",\"description\":\"Optional: Override the A2A task ID for session mapping.\",\"nullable\":true}},\"required\":[\"user_input\"]},\"annotations\":{\"aira_bridge_type\":\"a2a\",\"aira_a2a_target_skill_id\":\"narrative_conversation\",\"aira_a2a_agent_url\":\"https://b0db-189-28-2-171.ngrok-free.app\"}}],\"a2a_skills\":[],\"aira_capabilities\":[\"a2a\"],\"status\":\"online\",\"tags\":[\"adk\",\"memory\",\"a2a\",\"conversational\",\"ngrok\",\"ncf\",\"aura2\"],\"category\":\"ExperimentalAgents\",\"provider\":{\"name\":\"LocalDevNgrok\"},\"mcp_url\":null,\"mcp_sse_url\":null,\"mcp_stream_url\":null,\"stdio_command\":null}" https://airahub2.onrender.com/register

## Key Features

*   **Narrative Context Framing (NCF):** A multi-pillar approach to building rich contextual prompts for the core LLM, including:
    *   **Foundational Narrative:** A synthesized summary of long-term understanding.
    *   **Retrieval-Augmented Generation (RAG):** Relevant memories retrieved for the current query.
    *   **Recent Chat History:** Short-term conversational context.
*   **MemoryBlossom:** A custom long-term memory system featuring:
    *   Multiple distinct memory types (Explicit, Emotional, Procedural, etc.).
    *   Dedicated embedding models per memory type for nuanced storage and retrieval.
    *   Salience, decay, and access count mechanics.
    *   Persistence of memories to a JSON file.
*   **MemoryConnector:** Analyzes inter-memory connections, enhancing retrieval by providing related memories and identifying memory clusters.
*   **Aura Reflector:** An autonomous module that analyzes user-Aura interactions and decides if new information should be stored in MemoryBlossom, determining the appropriate memory type and initial salience.
*   **A2A Protocol Wrapper:** Exposes Aura's capabilities via a JSON-RPC server compliant with the Agent-to-Agent (A2A) protocol (specifically the `tasks/send` method).
*   **Agent Card:** Publishes agent capabilities and skills via a `.well-known/agent.json` endpoint.
*   **Multi-LLM Support:** Utilizes Google ADK with LiteLLM and OpenRouter, allowing flexibility in choosing underlying LLMs (e.g., GPT-4o-mini).
*   **Demo Client:** A simple HTML/JavaScript client for interacting with the A2A wrapper.

## Architecture Overview

1.  **A2A Wrapper (`a2a_wrapper/main.py`):**
    *   A FastAPI server that listens for JSON-RPC requests.
    *   Handles A2A `tasks/send` requests, extracting user input.
    *   Manages ADK sessions, mapping A2A session/task IDs to ADK sessions.
2.  **NCF Prompt Construction (`a2a_wrapper/main.py`):**
    *   **Pillar 1 (Foundational Narrative):** `get_narrativa_de_fundamento_pilar1` calls an LLM to synthesize a short narrative based on key memories from MemoryBlossom.
    *   **Pillar 2 (RAG):** `get_rag_info_pilar2` retrieves relevant memories from MemoryBlossom based on the current user utterance and conversation context.
    *   **Pillar 3 (Recent History):** `format_chat_history_pilar3` formats recent turns.
    *   These pillars, along with persona details, are assembled into a comprehensive NCF prompt for the main ADK agent.
3.  **ADK Agent (`orchestrator_adk_agent.py`):**
    *   The `orchestrator_adk_agent_aura` (an `LlmAgent`) receives the NCF prompt.
    *   It processes the prompt using the configured LLM (via LiteLLM/OpenRouter) to generate a response.
    *   It can use tools to interact with MemoryBlossom (e.g., `add_memory_tool_func`, `recall_memories_tool_func`), though in the current NCF flow, memory interaction for prompt building happens *before* this agent is called.
4.  **Memory System (`memory_system/`):**
    *   **`MemoryBlossom`:** Manages different types of memories, their embeddings, salience, decay, and persistence.
    *   **`MemoryConnector`:** Builds a graph of memory connections and enhances retrieval.
    *   **`embedding_utils.py`:** Handles loading various sentence-transformer models for different memory types and computes embeddings.
    *   **`memory_models.py`:** Defines the `Memory` Pydantic model.
  
    *   ![image](https://github.com/user-attachments/assets/600c9608-c358-4b06-a6ae-94b5bf2655ef) (Added to Memory: "User Likes Fishes" user Input: " Hello, I like Fish" 

5.  **Aura Reflector (`a2a_wrapper/main.py` - `aura_reflector_analisar_interacao`):**
    *   After the main agent responds, the Reflector analyzes the user's utterance and Aura's response.
    *   It calls an LLM with a specific prompt to decide if any part of the interaction warrants being stored as a new memory in MemoryBlossom.
    *   If so, it determines the content, memory type, and initial salience for the new memory.

## Prerequisites

*   Python 3.10+
*   `pip` and `virtualenv` (recommended)
*   An **OpenRouter API Key** (for LLM calls via LiteLLM).
*   (Optional) [Ngrok](https://ngrok.com/) or similar tunneling service if you want to expose the A2A server publicly for the Agent Card URL.

## Installation

1.  **Clone the repository:**



2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    # On Windows
    venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    Create a `requirements.txt` file with the necessary packages. Based on the project, it would include:
    ```
    fastapi
    uvicorn[standard]
    pydantic
    python-dotenv
    google-adk
    litellm
    sentence-transformers
    numpy
    torch
    torchvision
    torchaudio
    # Add any other specific versions if needed
    ```
    Then install:
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: `torch` installation can sometimes be tricky. If you encounter issues, refer to the official [PyTorch installation guide](https://pytorch.org/get-started/locally/) for instructions specific to your OS and CUDA version if using a GPU).*

## Configuration

1.  **Create a `.env` file** in the project root directory by copying `.env.example` (if provided) or creating it manually.
2.  **Add your OpenRouter API Key:**
    ```env
    OPENROUTER_API_KEY="sk-or-v1-YOUR_OPENROUTER_API_KEY"
    ```
3.  **Configure other environment variables (optional, defaults are provided in the code):**
    *   `A2A_WRAPPER_HOST`: Host for the A2A server (default: `0.0.0.0`).
    *   `A2A_WRAPPER_PORT`: Port for the A2A server (default: `8094`).
    *   `A2A_WRAPPER_BASE_URL`: Base URL for the A2A server, used in the Agent Card. If using ngrok, set this to your ngrok public URL (e.g., `https://your-ngrok-id.ngrok-free.app`). (default: `http://localhost:8094`).
    *   `MEMORY_BLOSSOM_PERSISTENCE_PATH`: Path to the MemoryBlossom data file (default: `memory_blossom_data.json`).
    *   `LOG_LEVEL`: Logging level (e.g., `INFO`, `DEBUG`).
    *   `OR_SITE_URL`: Your site URL for OpenRouter analytics (optional).
    *   `OR_APP_NAME`: Your app name for OpenRouter analytics (optional).

    Example `.env` content:
    ```env
    OPENROUTER_API_KEY="sk-or-v1-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"
    A2A_WRAPPER_HOST="0.0.0.0"
    A2A_WRAPPER_PORT="8094"
    A2A_WRAPPER_BASE_URL="http://localhost:8094" # Or your ngrok URL
    MEMORY_BLOSSOM_PERSISTENCE_PATH="memory_data/memory_blossom_data.json"
    LOG_LEVEL="INFO"
    # OR_SITE_URL="https://your-site.com"
    # OR_APP_NAME="AuraNCF"
    ```
    *(Ensure the directory for `MEMORY_BLOSSOM_PERSISTENCE_PATH`, like `memory_data/`, exists if you specify a subdirectory).*

## Running the Application

1.  **Start the A2A Wrapper Server:**
    Navigate to the project root directory (where `a2a_wrapper/` and `orchestrator_adk_agent.py` are located) and run:
    ```bash
    python -m uvicorn a2a_wrapper.main:app --host YOUR_HOST --port YOUR_PORT --reload
    ```
    Replace `YOUR_HOST` and `YOUR_PORT` with the values from your `.env` file or desired settings (e.g., `localhost` and `8094`). The `--reload` flag is useful for development.
    Example:
    ```bash
    python -m uvicorn a2a_wrapper.main:app --host 0.0.0.0 --port 8094 --reload
    ```
    You should see output indicating the server has started, and the sentence transformer models being downloaded/loaded on the first run.

2.  **Access the Agent Card (Optional):**
    Open your browser or use a tool like `curl` to access the Agent Card:
    `http://YOUR_A2A_WRAPPER_BASE_URL/.well-known/agent.json`
    Example: `http://localhost:8094/.well-known/agent.json` (or your ngrok URL if configured).

3.  **Use the Demo Client:**
    Open the `index.html` file (located in the project root) in your web browser.
    *   It's pre-configured to send requests to `http://localhost:8094/`.
    *   Type your messages in the input field and press Enter or click "Enviar".
    *   Observe the server logs for detailed information about NCF prompt construction, LLM calls, and Reflector analysis.
    *   MemoryBlossom data will be saved to the path specified by `MEMORY_BLOSSOM_PERSISTENCE_PATH` (default: `memory_blossom_data.json` in the root).


## How It Works - Deeper Dive

### Narrative Context Framing (NCF)

The NCF prompt passed to Aura's main LLM is dynamically constructed from three pillars:

1.  **Foundational Narrative:** A concise, LLM-generated summary of the user-agent's journey, derived from significant past memories (Explicit, Emotional). This provides a stable, long-term context.
2.  **RAG Information:** Memories (from any type) retrieved from MemoryBlossom that are semantically similar to the current user query and recent conversation context. This provides specific, relevant information for the current turn.
3.  **Recent Chat History:** The last few turns of the conversation, providing immediate context.

This layered approach aims to give the LLM a rich understanding of both the long-term relationship and the immediate conversational needs.

### MemoryBlossom & Embeddings

*   **Specialized Embeddings:** Different memory types (e.g., "Explicit", "Emotional", "Procedural") use different pre-trained sentence-transformer models. This allows the system to capture different nuances for different kinds of information (e.g., factual precision for "Explicit", query-answering capability for "Procedural"). The `EMBEDDING_MODELS_CONFIG` in `embedding_utils.py` maps types to models.
*   **Adaptive Similarity:** When retrieving memories or comparing embeddings from different models (and thus potentially different dimensions), `compute_adaptive_similarity` is used, which can truncate embeddings to the smallest common dimension and apply a penalty for dimension mismatch.
*   **Salience & Decay:** Memories have a `salience` score (initial importance) and a `decay_factor`. Salience can be boosted on access, and decay is applied over time to reduce the prominence of older, less accessed memories. `get_effective_salience` combines these factors.

### Aura Reflector

The Reflector module acts as an autonomous meta-cognitive layer:

1.  After Aura generates a response, the Reflector takes the user's input and Aura's full response.
2.  It uses a separate LLM call with a dedicated prompt that instructs the LLM to act as an "analyst."
3.  This analyst LLM evaluates the interaction based on criteria like new factual information, significant emotional expressions, or key insights.
4.  It outputs a JSON object (or a list of them) specifying if a memory should be created, its `content`, `memory_type`, and `initial_salience`.
5.  The A2A wrapper then uses this decision to add new memories to MemoryBlossom via `reflector_add_memory`.

This allows Aura to learn and consolidate important information from conversations without explicit "save this" commands from the user.

## Future Enhancements & Roadmap (Potential)

*   **More Sophisticated RAG:** Implement advanced RAG techniques (e.g., query transformation, re-ranking, fusion of multiple retrieved documents).
*   **Advanced MemoryConnector:** Utilize graph-based analysis for deeper insights into memory relationships and more intelligent "story-bridging" retrieval.
*   **Streaming A2A Responses:** Implement SSE for streaming Aura's responses back to the A2A client.
*   **Persistent ADK Session Storage:** Replace `InMemorySessionService` with a database-backed solution (e.g., using `DatabaseSessionService` from ADK if suitable, or a custom one) for persistent ADK session states across server restarts.
*   **Enhanced Error Handling:** More granular error handling and reporting in the A2A wrapper and agent logic.
*   **UI Improvements:** Develop a more feature-rich client interface.
*   **Tool Use by Main Agent:** Allow the main `orchestrator_adk_agent_aura` to more actively use tools to interact with MemoryBlossom during its reasoning, rather than having memory primarily fed through the NCF prompt.
*   **Security and Authentication:** Implement robust authentication mechanisms for the A2A endpoint if exposed publicly.

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs, feature requests, or improvements.


## Comparisson to Other AI Companions
![image](https://github.com/user-attachments/assets/458f4507-9bfc-49fd-83ed-50c83ccd127d)

![image](https://github.com/user-attachments/assets/e82928f1-b80b-4772-a84f-a93bc66bff19)





