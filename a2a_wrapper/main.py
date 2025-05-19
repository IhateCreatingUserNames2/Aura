# a2a_wrapper/main.py
import uvicorn
from fastapi import FastAPI, Request as FastAPIRequest
from fastapi.responses import JSONResponse
import json
import uuid
from datetime import datetime, timezone
from typing import Union, Dict, Any, List, Optional
import logging
import os
from dotenv import load_dotenv
from types import SimpleNamespace # +++ Add this import +++


# For LLM calls within pillar/reflector functions
from google.adk.models.lite_llm import LiteLlm
from starlette.middleware.cors import CORSMiddleware
from google.adk.models.llm_request import LlmRequest
# from google.adk.models.model_config import ModelConfig # Keep this commented or remove


# from google.genai.types import GenerateContentConfig # Only if explicitly configuring helper_llm

# --- Load .env file ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
dotenv_path = os.path.join(PROJECT_ROOT, '.env')

if os.path.exists(dotenv_path):
    print(f"A2A Wrapper: Loading .env file from: {dotenv_path}")
    load_dotenv(dotenv_path)
else:
    print(f"A2A Wrapper: .env file not found at {dotenv_path}. Relying on environment variables.")

# --- Module Imports ---
from orchestrator_adk_agent import (
    adk_runner,
    orchestrator_adk_agent_aura,
    ADK_APP_NAME,
    memory_blossom_instance,
    reflector_add_memory,
    AGENT_MODEL_STRING
)
from memory_system.memory_blossom import MemoryBlossom
from memory_system.memory_models import Memory as MemoryModel

from a2a_wrapper.models import (
    A2APart, A2AMessage, A2ATaskSendParams, A2AArtifact,
    A2ATaskStatus, A2ATaskResult, A2AJsonRpcRequest, A2AJsonRpcResponse,
    AgentCard, AgentCardSkill, AgentCardProvider, AgentCardAuthentication, AgentCardCapabilities
)

from google.genai.types import Content as ADKContent  # Used for constructing messages
from google.genai.types import Part as ADKPart  # Used for constructing messages
from google.adk.sessions import Session as ADKSession

logger = logging.getLogger(__name__)
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper())

# --- Configuration & FastAPI App ---
A2A_WRAPPER_HOST = os.getenv("A2A_WRAPPER_HOST", "0.0.0.0")
A2A_WRAPPER_PORT = int(os.getenv("A2A_WRAPPER_PORT", "8094"))
A2A_WRAPPER_BASE_URL = os.getenv("A2A_WRAPPER_BASE_URL", f"http://localhost:{A2A_WRAPPER_PORT}")

app = FastAPI(
    title="Aura Agent A2A Wrapper (NCF Prototype v1.2)",
    description="Exposes the Aura (NCF) ADK agent via the A2A protocol, with multi-agent inspired NCF prompt building."
)

# --- Configuração do CORS ---
origins = [
    "http://localhost",
    "http://localhost:8000",
    "http://127.0.0.1",
    "http://127.0.0.1:8000",
    "null",
]
origins = [origin for origin in origins if origin]
if not origins:
    origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Agent Card ---
AGENT_CARD_DATA = AgentCard(
    name="Aura",
    description="A conversational AI agent, Aura, with advanced memory capabilities using "
                "Narrative Context Framing (NCF). Aura aims to build a deep, "
                "contextual understanding over long interactions.",
    url=A2A_WRAPPER_BASE_URL,
    version="1.2.0-ncf-proto",
    provider=AgentCardProvider(organization="LocalDev", url=os.environ.get("OR_SITE_URL", "http://example.com")),
    capabilities=AgentCardCapabilities(streaming=False, pushNotifications=False),
    authentication=AgentCardAuthentication(schemes=[]),
    skills=[
        AgentCardSkill(
            id="narrative_conversation",
            name="Narrative Conversation with Aura",
            description="Engage in a deep, contextual conversation. Aura uses its "
                        "MemoryBlossom system and Narrative Context Framing to understand "
                        "and build upon previous interactions.",
            tags=["chat", "conversation", "memory", "ncf", "context", "aura", "multi-agent-concept"],
            examples=[
                "Let's continue our discussion about emergent narrative structures.",
                "Based on what we talked about regarding liminal spaces, what do you think about this new idea?",
                "Store this feeling: 'The breakthrough in my research felt incredibly liberating!'",
                "How does our previous conversation on AI ethics relate to this new scenario?"
            ],
            parameters={
                "type": "object",
                "properties": {
                    "user_input": {
                        "type": "string",
                        "description": "The textual input from the user for the conversation."
                    },
                    "a2a_task_id_override": {
                        "type": "string",
                        "description": "Optional: Override the A2A task ID for session mapping.",
                        "nullable": True
                    }
                },
                "required": ["user_input"]
            }
        )
    ]
)


@app.get("/.well-known/agent.json", response_model=AgentCard, response_model_exclude_none=True)
async def get_agent_card():
    return AGENT_CARD_DATA


a2a_task_to_adk_session_map: Dict[str, str] = {}
helper_llm = LiteLlm(model=AGENT_MODEL_STRING)


# --- Aura NCF Pillar Functions ---

async def get_narrativa_de_fundamento_pilar1(
        current_session_state: Dict[str, Any],
        mb_instance: MemoryBlossom,
        user_id: str
) -> str:
    logger.info(f"[Pilar 1] Generating Narrative Foundation for user {user_id}...")
    if 'foundation_narrative' in current_session_state and current_session_state.get('foundation_narrative_turn_count',
                                                                                     0) < 5:
        current_session_state['foundation_narrative_turn_count'] = current_session_state.get(
            'foundation_narrative_turn_count', 0) + 1
        logger.info(
            f"[Pilar 1] Using cached Narrative Foundation. Turn count: {current_session_state['foundation_narrative_turn_count']}")
        return current_session_state['foundation_narrative']

    relevant_memories_for_foundation: List[MemoryModel] = []
    try:
        explicit_mems = mb_instance.retrieve_memories(
            query="key explicit facts and statements from our past discussions", top_k=2,
            target_memory_types=["Explicit"], apply_criticality=False)
        emotional_mems = mb_instance.retrieve_memories(query="significant emotional moments or sentiments expressed",
                                                       top_k=1, target_memory_types=["Emotional"],
                                                       apply_criticality=False)
        relevant_memories_for_foundation.extend(explicit_mems)
        relevant_memories_for_foundation.extend(emotional_mems)
        seen_ids = set()
        unique_memories = []
        for mem in relevant_memories_for_foundation:
            if mem.id not in seen_ids:
                unique_memories.append(mem)
                seen_ids.add(mem.id)
        relevant_memories_for_foundation = unique_memories
    except Exception as e:
        logger.error(f"[Pilar 1] Error retrieving memories for foundation: {e}", exc_info=True)
        return "Estamos construindo nossa jornada de entendimento mútuo."

    if not relevant_memories_for_foundation:
        narrative = "Nossa jornada de aprendizado e descoberta está apenas começando. Estou ansiosa para explorar vários tópicos interessantes com você."
    else:
        memory_contents = [f"- ({mem.memory_type}): {mem.content}" for mem in relevant_memories_for_foundation]
        memories_str = "\n".join(memory_contents)
        synthesis_prompt = f"""
        Você é um sintetizador de narrativas. Com base nas seguintes memórias chave de interações passadas, crie uma breve narrativa de fundamento (1-2 frases concisas) que capture a essência da nossa jornada de entendimento e os principais temas discutidos. Esta narrativa servirá como pano de fundo para nossa conversa atual.

        Memórias Chave:
        {memories_str}

        Narrativa de Fundamento Sintetizada:
        """
        try:
            logger.info(
                f"[Pilar 1] Calling LLM to synthesize Narrative Foundation from {len(relevant_memories_for_foundation)} memories.")
            request_messages = [ADKContent(parts=[ADKPart(text=synthesis_prompt)])]
            # +++ Use SimpleNamespace for config +++
            # This creates an object for `req.config` with a `tools` attribute that is an empty list.
            minimal_config = SimpleNamespace(tools=[])
            llm_req = LlmRequest(contents=request_messages, config=minimal_config)


            final_text_response = ""
            async for llm_response_event in helper_llm.generate_content_async(
                    llm_req):
                if llm_response_event and llm_response_event.content and \
                        llm_response_event.content.parts and llm_response_event.content.parts[0].text:
                    final_text_response += llm_response_event.content.parts[0].text

            narrative = final_text_response.strip() if final_text_response else "Continuamos a construir nossa compreensão mútua com base em nossas interações anteriores."
        except Exception as e:
            logger.error(f"[Pilar 1] LLM error synthesizing Narrative Foundation: {e}", exc_info=True)
            narrative = "Refletindo sobre nossas conversas anteriores para guiar nosso diálogo atual."

    current_session_state['foundation_narrative'] = narrative
    current_session_state['foundation_narrative_turn_count'] = 1
    logger.info(f"[Pilar 1] Generated new Narrative Foundation: '{narrative[:100]}...'")
    return narrative


async def get_rag_info_pilar2(
        user_utterance: str,
        mb_instance: MemoryBlossom,
        current_session_state: Dict[str, Any]
) -> List[Dict[str, Any]]:
    logger.info(f"[Pilar 2] Retrieving RAG info for utterance: '{user_utterance[:50]}...'")
    try:
        conversation_context = current_session_state.get('conversation_history', [])[-5:]
        recalled_memories_for_rag = mb_instance.retrieve_memories(
            query=user_utterance, top_k=3, conversation_context=conversation_context
        )
        rag_results = [mem.to_dict() for mem in recalled_memories_for_rag]
        logger.info(f"[Pilar 2] Retrieved {len(rag_results)} memories for RAG.")
        return rag_results
    except Exception as e:
        logger.error(f"[Pilar 2] Error in get_rag_info: {e}", exc_info=True)
        return [{"content": f"Erro ao buscar informações RAG específicas: {str(e)}", "memory_type": "Error"}]


def format_chat_history_pilar3(chat_history_list: List[Dict[str, str]], max_turns: int = 15) -> str:
    if not chat_history_list: return "Nenhum histórico de conversa recente disponível."
    recent_history = chat_history_list[-max_turns:]
    formatted_history = [f"{'Usuário' if entry.get('role') == 'user' else 'Aura'}: {entry.get('content')}" for entry in
                         recent_history]
    return "\n".join(
        formatted_history) if formatted_history else "Nenhum histórico de conversa recente disponível para formatar."


def montar_prompt_aura_ncf(
        persona_agente: str, persona_detalhada: str, narrativa_fundamento: str,
        informacoes_rag_list: List[Dict[str, Any]], chat_history_recente_str: str, user_reply: str
) -> str:
    logger.info("[PromptBuilder] Assembling NCF prompt...")
    formatted_rag = ""
    if informacoes_rag_list:
        rag_items_str = [
            f"  - ({item_dict.get('memory_type', 'Info')}): {item_dict.get('content', 'Conteúdo indisponível')}"
            for item_dict in informacoes_rag_list
        ]
        formatted_rag = "Informações e memórias específicas que podem ser úteis para esta interação (RAG):\n" + "\n".join(
            rag_items_str) if rag_items_str \
            else "Nenhuma informação específica (RAG) foi recuperada para esta consulta."
    else:
        formatted_rag = "Nenhuma informação específica (RAG) foi recuperada para esta consulta."

    task_instruction = """## Sua Tarefa:
Responda ao usuário de forma natural, coerente e útil, levando em consideração TODA a narrativa de contexto e o histórico fornecido.
- Incorpore ativamente elementos da "Narrativa de Fundamento" para mostrar continuidade e entendimento profundo.
- Utilize as "Informações RAG" para embasar respostas específicas ou fornecer detalhes relevantes.
- Mantenha a persona definida.
- Se identificar uma aparente contradição entre a "Narrativa de Fundamento", as "Informações RAG" ou o "Histórico Recente", tente abordá-la com humildade epistêmica:
    - Priorize a informação mais recente ou específica, se aplicável.
    - Considere se é uma evolução do entendimento ou um novo aspecto.
    - Se necessário, você pode mencionar sutilmente a aparente diferença ou pedir clarificação ao usuário de forma implícita através da sua resposta. Não afirme categoricamente que há uma contradição, mas navegue a informação com nuance.
- Evite redundância. Se o histórico recente já cobre um ponto, não o repita extensivamente a menos que seja para reforçar uma conexão crucial com a nova informação.
"""
    prompt = f"""<SYSTEM_PERSONA_START>
Você é {persona_agente}.
{persona_detalhada}
<SYSTEM_PERSONA_END>

<NARRATIVE_FOUNDATION_START>
## Nosso Entendimento e Jornada Até Agora (Narrativa de Fundamento):
{narrativa_fundamento if narrativa_fundamento else "Ainda não construímos uma narrativa de fundamento detalhada para nossa interação."}
<NARRATIVE_FOUNDATION_END>

<SPECIFIC_CONTEXT_RAG_START>
## Informações Relevantes para a Conversa Atual (RAG):
{formatted_rag}
<SPECIFIC_CONTEXT_RAG_END>

<RECENT_HISTORY_START>
## Histórico Recente da Nossa Conversa:
{chat_history_recente_str if chat_history_recente_str else "Não há histórico recente disponível para esta conversa."}
<RECENT_HISTORY_END>

<CURRENT_SITUATION_START>
## Situação Atual:
Você está conversando com o usuário. O usuário acabou de dizer:

Usuário: "{user_reply}"

{task_instruction}
<CURRENT_SITUATION_END>

Aura:"""
    logger.info(f"[PromptBuilder] NCF Prompt assembled. Length: {len(prompt)}")
    return prompt


async def aura_reflector_analisar_interacao(
        user_utterance: str, prompt_ncf_usado: str, resposta_de_aura: str,
        mb_instance: MemoryBlossom, user_id: str
):
    logger.info(f"[Reflector] Analisando interação para user {user_id}...")
    logger.debug(
        f"[Reflector] Interaction Log for {user_id}:\nUser Utterance: {user_utterance}\nNCF Prompt (first 500): {prompt_ncf_usado[:500]}...\nAura's Resp: {resposta_de_aura}")

    reflector_prompt = f"""
    Você é um analista de conversas de IA chamado "Reflector". Sua tarefa é analisar a seguinte interação entre um usuário e a IA Aura para identificar se alguma informação crucial deve ser armazenada na memória de longo prazo de Aura (MemoryBlossom).
    Contexto da IA Aura: Aura usa uma Narrativa de Fundamento (resumo de longo prazo), RAG (informações específicas para a query) e Histórico Recente para responder.
    O prompt completo que Aura recebeu já continha muito desse contexto.
    Agora, avalie a *nova* informação trocada (pergunta do usuário e resposta de Aura) e decida se algo dessa *nova troca* merece ser uma memória distinta.
    Critérios para decidir armazenar uma memória:
    1.  **Fatos explícitos importantes declarados pelo usuário ou pela Aura** que provavelmente serão relevantes no futuro (ex: preferências do usuário, decisões chave, novas informações factuais significativas que Aura aprendeu ou ensinou).
    2.  **Momentos emocionais significativos** expressos pelo usuário ou refletidos por Aura que indicam um ponto importante na interação.
    3.  **Insights ou conclusões chave** alcançados durante a conversa.
    4.  **Correções importantes feitas pelo usuário e aceitas por Aura.**
    5.  **Tarefas ou objetivos de longo prazo** mencionados.
    NÃO armazene:
    - Conversa trivial, saudações, despedidas (a menos que contenham emoção significativa).
    - Informação que já está claramente coberta pela Narrativa de Fundamento ou RAG que foi fornecida a Aura (a menos que a interação atual adicione um novo significado ou conexão a ela).
    - Perguntas do usuário, a menos que a pergunta em si revele uma nova intenção de longo prazo ou um fato sobre o usuário.
    Interação para Análise:
    Usuário disse: "{user_utterance}"
    Aura respondeu: "{resposta_de_aura}"
    Com base na sua análise, se você acha que uma ou mais memórias devem ser criadas, forneça a resposta no seguinte formato JSON. Se múltiplas memórias, forneça uma lista de objetos JSON. Se nada deve ser armazenado, retorne um JSON vazio `{{}}` ou uma lista vazia `[]`.
    Formato JSON para cada memória a ser criada:
    {{
      "content": "O conteúdo textual da memória a ser armazenada. Seja conciso mas completo.",
      "memory_type": "Escolha um de: Explicit, Emotional, Procedural, Flashbulb, Liminal, Generative",
      "emotion_score": 0.0-1.0 (se memory_type for Emotional, senão pode ser 0.0),
      "initial_salience": 0.0-1.0 (quão importante parece ser esta memória? 0.5 é neutro, 0.8 é importante),
      "metadata": {{ "source": "aura_reflector_analysis", "user_id": "{user_id}", "related_interaction_turn": "current" }}
    }}
    Sua decisão (JSON):
    """
    try:
        logger.info(f"[Reflector] Chamando LLM para decisão de armazenamento de memória.")
        request_messages = [ADKContent(parts=[ADKPart(text=reflector_prompt)])]
        # +++ Use SimpleNamespace for config +++
        minimal_config = SimpleNamespace(tools=[])
        llm_req = LlmRequest(contents=request_messages, config=minimal_config)


        final_text_response = ""
        async for llm_response_event in helper_llm.generate_content_async(llm_req):
            if llm_response_event and llm_response_event.content and \
                    llm_response_event.content.parts and llm_response_event.content.parts[0].text:
                final_text_response += llm_response_event.content.parts[0].text

        if not final_text_response:
            logger.info("[Reflector] Nenhuma decisão de armazenamento de memória retornada pelo LLM.")
            return

        decision_json_str = final_text_response.strip()
        logger.info(f"[Reflector] Decisão de armazenamento (JSON string): {decision_json_str}")

        if '```json' in decision_json_str:
            decision_json_str = decision_json_str.split('```json')[1].split('```')[0].strip()
        elif not (decision_json_str.startswith('{') and decision_json_str.endswith('}')) and \
                not (decision_json_str.startswith('[') and decision_json_str.endswith(']')):
            match_obj, match_list = None, None
            try:
                obj_start = decision_json_str.index('{'); obj_end = decision_json_str.rindex('}') + 1
                match_obj = decision_json_str[obj_start:obj_end]
            except ValueError: pass
            try:
                list_start = decision_json_str.index('['); list_end = decision_json_str.rindex(']') + 1
                match_list = decision_json_str[list_start:list_end]
            except ValueError: pass
            if match_obj and (not match_list or len(match_obj) > len(match_list)): decision_json_str = match_obj
            elif match_list: decision_json_str = match_list
            else:
                logger.warning(f"[Reflector] LLM response not valid JSON after cleaning: {decision_json_str}")
                return

        memories_to_add = []
        try:
            parsed_decision = json.loads(decision_json_str)
            if isinstance(parsed_decision, dict) and "content" in parsed_decision and "memory_type" in parsed_decision:
                memories_to_add.append(parsed_decision)
            elif isinstance(parsed_decision, list):
                memories_to_add = [item for item in parsed_decision if isinstance(item, dict) and "content" in item and "memory_type" in item]
            elif parsed_decision: # Non-empty but not valid memory structure
                 logger.info(f"[Reflector] Parsed decision is not a valid memory structure: {parsed_decision}")

        except json.JSONDecodeError as e:
            logger.error(f"[Reflector] JSONDecodeError for Reflector decision: {e}. String: {decision_json_str}")
            return

        for mem_data in memories_to_add:
            logger.info(
                f"[Reflector] Adding memory: Type='{mem_data['memory_type']}', Content='{mem_data['content'][:50]}...'")
            reflector_add_memory(
                content=mem_data["content"], memory_type=mem_data["memory_type"],
                emotion_score=float(mem_data.get("emotion_score", 0.0)),
                initial_salience=float(mem_data.get("initial_salience", 0.5)),
                metadata=mem_data.get("metadata", {"source": "aura_reflector_analysis", "user_id": user_id})
            )
    except Exception as e:
        logger.error(f"[Reflector] Error during interaction analysis: {e}", exc_info=True)


# --- A2A RPC Handler ---
@app.post("/", response_model=A2AJsonRpcResponse, response_model_exclude_none=True)
async def handle_a2a_rpc(rpc_request: A2AJsonRpcRequest, http_request: FastAPIRequest):
    client_host = http_request.client.host if http_request.client else "unknown"
    logger.info(
        f"\nA2A Wrapper: Received request from {client_host}: Method={rpc_request.method}, RPC_ID={rpc_request.id}")

    if rpc_request.method == "tasks/send":
        if rpc_request.params is None:
            logger.error("A2A Wrapper: Error - Missing 'params' for tasks/send")
            return A2AJsonRpcResponse(id=rpc_request.id, error={"code": -32602, "message": "Invalid params: missing"})
        try:
            task_params = rpc_request.params
            logger.info(f"A2A Wrapper: Processing tasks/send for A2A Task ID: {task_params.id}")

            user_utterance_raw = ""
            if task_params.message and task_params.message.parts:
                first_part = task_params.message.parts[0]
                if first_part.type == "data" and first_part.data and "user_input" in first_part.data:
                    user_utterance_raw = first_part.data["user_input"]
                elif first_part.type == "text" and first_part.text:
                    user_utterance_raw = first_part.text

            if not user_utterance_raw:
                logger.error("A2A Wrapper: Error - No user_input found in A2A message.")
                return A2AJsonRpcResponse(id=rpc_request.id,
                                          error={"code": -32602, "message": "Invalid params: user_input missing"})
            logger.info(f"A2A Wrapper: Extracted user_utterance_raw: '{user_utterance_raw[:50]}...'")

            adk_session_key_override = None
            if task_params.message.parts[0].data and task_params.message.parts[0].data.get("a2a_task_id_override"):
                adk_session_key_override = task_params.message.parts[0].data["a2a_task_id_override"]
                logger.info(
                    f"A2A Wrapper: Using overridden A2A Task ID for ADK session mapping: {adk_session_key_override}")

            adk_session_map_key = adk_session_key_override or task_params.sessionId or task_params.id
            adk_user_id = f"a2a_user_for_{adk_session_map_key}"
            adk_session_id = a2a_task_to_adk_session_map.get(adk_session_map_key)

            current_adk_session: Optional[ADKSession] = None
            if adk_session_id:
                current_adk_session = adk_runner.session_service.get_session(ADK_APP_NAME, adk_user_id, adk_session_id)

            if not current_adk_session:
                adk_session_id = f"adk_session_for_{adk_session_map_key}_{str(uuid.uuid4())[:8]}"
                a2a_task_to_adk_session_map[adk_session_map_key] = adk_session_id
                current_adk_session = adk_runner.session_service.create_session(
                    app_name=ADK_APP_NAME,
                    user_id=adk_user_id,
                    session_id=adk_session_id,
                    state={'conversation_history': [], 'foundation_narrative_turn_count': 0}
                )
                logger.info(
                    f"A2A Wrapper: Created ADK session '{adk_session_id}' (mapped from '{adk_session_map_key}') for user '{adk_user_id}'")
            else:
                logger.info(
                    f"A2A Wrapper: Reusing ADK session '{adk_session_id}' (mapped from '{adk_session_map_key}') for user '{adk_user_id}'")

            if 'conversation_history' not in current_adk_session.state: current_adk_session.state['conversation_history'] = []
            if 'foundation_narrative_turn_count' not in current_adk_session.state: current_adk_session.state['foundation_narrative_turn_count'] = 0

            current_adk_session.state['conversation_history'].append({"role": "user", "content": user_utterance_raw})

            logger.info(f"--- Starting NCF Prompt Construction for User: {adk_user_id}, Session: {adk_session_id} ---")
            narrativa_fundamento = await get_narrativa_de_fundamento_pilar1(current_adk_session.state, memory_blossom_instance, adk_user_id)
            rag_info_list = await get_rag_info_pilar2(user_utterance_raw, memory_blossom_instance, current_adk_session.state)
            chat_history_for_prompt_str = format_chat_history_pilar3(current_adk_session.state['conversation_history'])

            aura_persona_agente = "Aura"
            aura_persona_detalhada = "Você é uma IA conversacional avançada, projetada para engajar em diálogos profundos e contextuais. Sua arquitetura é inspirada em conceitos de coerência narrativa e humildade epistêmica, buscando construir um entendimento contínuo com o usuário. Você se esforça para ser perspicaz, adaptável e consciente das nuances da conversa."

            final_ncf_prompt_str = montar_prompt_aura_ncf(
                aura_persona_agente, aura_persona_detalhada, narrativa_fundamento,
                rag_info_list, chat_history_for_prompt_str, user_utterance_raw
            )
            logger.debug(f"A2A Wrapper: NCF Prompt (first 500 chars):\n{final_ncf_prompt_str[:500]}...")

            adk_input_content = ADKContent(role="user", parts=[ADKPart(text=final_ncf_prompt_str)])
            logger.info(f"A2A Wrapper: Running Aura ADK agent for session '{adk_session_id}'")
            adk_agent_final_text_response = None

            async for event in adk_runner.run_async(
                    user_id=adk_user_id, session_id=adk_session_id, new_message=adk_input_content
            ):
                logger.debug(f"  ADK Event: Author={event.author}, Final={event.is_final_response()}, Content Present={bool(event.content)}")
                if event.get_function_calls():
                    fc = event.get_function_calls()[0]
                    logger.info(f"    ADK FunctionCall by {event.author}: {fc.name}({json.dumps(fc.args)})")
                if event.get_function_responses():
                    fr = event.get_function_responses()[0]
                    logger.info(f"    ADK FunctionResponse to {event.author}: {fr.name} -> {str(fr.response)[:100]}...")
                if event.is_final_response():
                    if event.content and event.content.parts and event.content.parts[0].text:
                        adk_agent_final_text_response = event.content.parts[0].text.strip()
                        logger.info(f"  Aura ADK Final Response Text: '{adk_agent_final_text_response[:100]}...'")
                    break

            adk_agent_final_text_response = adk_agent_final_text_response or "(Aura não forneceu uma resposta textual para este turno)"
            current_adk_session.state['conversation_history'].append({"role": "assistant", "content": adk_agent_final_text_response})

            await aura_reflector_analisar_interacao(
                user_utterance_raw, final_ncf_prompt_str, adk_agent_final_text_response,
                memory_blossom_instance, adk_user_id
            )

            a2a_response_artifact = A2AArtifact(parts=[A2APart(type="text", text=adk_agent_final_text_response)])
            a2a_task_status = A2ATaskStatus(state="completed")
            a2a_task_result = A2ATaskResult(
                id=task_params.id, sessionId=task_params.sessionId,
                status=a2a_task_status, artifacts=[a2a_response_artifact]
            )
            logger.info(f"A2A Wrapper: Sending A2A response for Task ID {task_params.id}")
            return A2AJsonRpcResponse(id=rpc_request.id, result=a2a_task_result)

        except ValueError as ve:
            logger.error(f"A2A Wrapper: Value Error: {ve}", exc_info=True)
            return A2AJsonRpcResponse(id=rpc_request.id, error={"code": -32602, "message": f"Invalid params: {ve}"})
        except Exception as e:
            logger.error(f"A2A Wrapper: Internal Error: {e}", exc_info=True)
            return A2AJsonRpcResponse(id=rpc_request.id, error={"code": -32000, "message": f"Internal Server Error: {e}"})
    else:
        logger.warning(f"A2A Wrapper: Method '{rpc_request.method}' not supported.")
        return A2AJsonRpcResponse(id=rpc_request.id, error={"code": -32601, "message": f"Method not found: {rpc_request.method}"})


if __name__ == "__main__":
    if not os.getenv("OPENROUTER_API_KEY"):
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print("!!! WARNING: OPENROUTER_API_KEY is not set. Agent will likely fail LLM calls. !!!")
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    logger.info(f"Starting Aura ADK A2A Wrapper Server on {A2A_WRAPPER_HOST}:{A2A_WRAPPER_PORT}")
    logger.info(f"Agent Card available at: {A2A_WRAPPER_BASE_URL}/.well-known/agent.json")
    uvicorn.run("main:app", host=A2A_WRAPPER_HOST, port=A2A_WRAPPER_PORT, reload=True, app_dir="a2a_wrapper")