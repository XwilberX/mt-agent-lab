"""Construcción del Agent PydanticAI + ejecutor.

El Agent se crea fresco por request (no hay estado compartido cross-tenant).
El modelo Ollama es local y stateless por request.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic_ai import Agent

from internal.agent.deps import AgentDeps
from internal.agent.prompt import build_system_prompt
from internal.agent.tools import consultar, describir_tabla, listar_tablas


DATA_ROOT = Path(os.environ.get("DATA_ROOT", "./data/parquet")).resolve()
# LLM_PROVIDER: "ollama" (default) usa OllamaProvider; "openai" usa OpenAIProvider
# para servidores OpenAI-compat puros como vLLM. base_url y model siguen leyéndose
# de OLLAMA_BASE_URL / OLLAMA_MODEL para no romper compat con scripts existentes.
LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "ollama")
OLLAMA_BASE_URL = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434/v1")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:14b")


def _build_model():
    """Construye el modelo según LLM_PROVIDER."""
    if LLM_PROVIDER == "openai":
        from pydantic_ai.models.openai import OpenAIChatModel
        from pydantic_ai.providers.openai import OpenAIProvider
        provider = OpenAIProvider(base_url=OLLAMA_BASE_URL, api_key="EMPTY")
        return OpenAIChatModel(model_name=OLLAMA_MODEL, provider=provider)

    from pydantic_ai.models.ollama import OllamaModel
    from pydantic_ai.providers.ollama import OllamaProvider
    provider = OllamaProvider(base_url=OLLAMA_BASE_URL)
    return OllamaModel(model_name=OLLAMA_MODEL, provider=provider)


def build_agent_for(tenant_slug: str, tenant_name: str, user_id: str) -> tuple[Agent, AgentDeps]:
    """Devuelve un Agent fresco + deps scoped al request.

    El system prompt se construye con `tenant_name` baked-in. No se reusa
    entre tenants.
    """
    deps = AgentDeps(
        tenant_slug=tenant_slug,
        tenant_name=tenant_name,
        data_root=DATA_ROOT,
        user_id=user_id,
    )

    agent = Agent(
        model=_build_model(),
        deps_type=AgentDeps,
        system_prompt=build_system_prompt(deps),
        retries=5,  # default era 1 → el modelo se trababa pidiendo describir_tabla
        model_settings={"temperature": 0.1},
    )

    agent.tool(listar_tablas)
    agent.tool(describir_tabla)
    agent.tool(consultar)

    return agent, deps


@dataclass
class AgentResult:
    answer: str
    tool_calls: list[dict[str, Any]]
    duration_ms: int
    all_messages_json: bytes  # serializado para persistir en conversations


async def run_agent(
    tenant_slug: str,
    tenant_name: str,
    user_id: str,
    question: str,
    message_history_json: bytes | None = None,
) -> AgentResult:
    """Ejecuta una vuelta del agente.

    Si se pasa `message_history_json`, lo deserializa y se lo da al Agent
    como contexto previo de la conversación.
    """
    import time

    from pydantic_ai.messages import ModelMessagesTypeAdapter

    agent, deps = build_agent_for(tenant_slug, tenant_name, user_id)

    msg_history = None
    if message_history_json:
        msg_history = ModelMessagesTypeAdapter.validate_json(message_history_json)

    t0 = time.monotonic()
    result = await agent.run(question, deps=deps, message_history=msg_history)
    duration_ms = int((time.monotonic() - t0) * 1000)

    # Extraer traza de tool calls de los nuevos messages
    tool_calls: list[dict[str, Any]] = []
    try:
        for msg in result.new_messages():
            for part in getattr(msg, "parts", []):
                if part.__class__.__name__ == "ToolCallPart":
                    tool_calls.append({
                        "tool": getattr(part, "tool_name", "?"),
                        "args": getattr(part, "args", None),
                    })
    except Exception:
        pass

    return AgentResult(
        answer=result.output,
        tool_calls=tool_calls,
        duration_ms=duration_ms,
        all_messages_json=result.all_messages_json(),
    )
