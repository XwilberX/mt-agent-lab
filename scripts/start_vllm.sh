#!/usr/bin/env bash
# Levanta vLLM-metal en el puerto 8001 con Mistral Small 24B (4-bit MLX)
# y configuración de tool calling validada en este PoC.
#
# Requisitos:
#   1. macOS Apple Silicon (M1/M2/M3/M4)
#   2. vllm-metal instalado en ~/.venv-vllm-metal
#      curl -fsSL https://raw.githubusercontent.com/vllm-project/vllm-metal/main/install.sh | bash
#
# El modelo se descarga la primera vez (~13GB) desde HuggingFace.
set -euo pipefail

MODEL="${VLLM_MODEL:-mlx-community/Mistral-Small-24B-Instruct-2501-4bit}"
PORT="${VLLM_PORT:-8001}"
TEMPLATE="$(dirname "$0")/tool_chat_template_mistral.jinja"

if [[ ! -d "$HOME/.venv-vllm-metal" ]]; then
  echo "✗ ~/.venv-vllm-metal no existe. Instalá con:"
  echo "  curl -fsSL https://raw.githubusercontent.com/vllm-project/vllm-metal/main/install.sh | bash"
  exit 1
fi

# shellcheck disable=SC1091
source "$HOME/.venv-vllm-metal/bin/activate"

echo "→ vLLM serve: $MODEL on :$PORT"
echo "→ tool-call-parser: mistral"
echo "→ chat-template: $TEMPLATE"

exec vllm serve "$MODEL" \
  --port "$PORT" \
  --enable-auto-tool-choice \
  --tool-call-parser mistral \
  --chat-template "$TEMPLATE"
