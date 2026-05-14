# Diagramas de arquitectura — propuesta para producción

Tres vistas de cómo desplegar el patrón validado en el PoC a una infra
**on-premise** real. La diferencia clave respecto al lab local:

| Componente | Lab (M2 Max) | Producción on-prem |
|---|---|---|
| LLM serving | vLLM-metal (MLX) | vLLM (CUDA) en GPU pool A100/L40S |
| Orquestación | docker-compose + uvicorn | Kubernetes (Rancher/OpenShift/kubeadm) |
| Postgres | container single-node | Patroni HA primary+replica |
| Object storage | filesystem `data/parquet/` | MinIO/Ceph cluster |
| Secrets | `.env` | HashiCorp Vault |
| Auth | header `X-User-Id` | Keycloak (OIDC) |
| Observability | structured logs a stderr | Prometheus + Grafana + Loki + Tempo |

## Archivos

| Archivo | Vista | Propósito |
|---|---|---|
| `01-deployment-topology.drawio` | Topología de despliegue | Quién corre dónde, qué red habla con qué, donde están los puntos de falla |
| `02-request-flow.drawio` | Flujo de un request | Una pregunta `POST /t/{slug}/ask` paso a paso, con los puntos de aislamiento marcados |
| `03-data-model.drawio` | Modelo de datos | Schemas Postgres por tenant + layout de MinIO + reglas de acceso |

## Cómo abrirlos

Son archivos `.drawio` nativos (XML mxGraph). Tres opciones:

1. **Online sin login** — [app.diagrams.net](https://app.diagrams.net) → File → Open from → Device. Edición y export sin cuenta.
2. **drawio Desktop** — descarga gratuita en [drawio.com](https://www.drawio.com/), recomendado para export a PNG/SVG/PDF.
3. **VS Code** — extensión `hediet.vscode-drawio` los abre inline.

## Decisiones abiertas (no fijadas en los diagramas)

- **GPU hardware:** A100 80GB single (Mistral 24B fp16 + 32k KV cache entra) vs L40S TP=2 (más barato, menos VRAM por shard). Depende de presupuesto y proveedor.
- **Postgres operator:** Patroni manual vs CloudNativePG vs Zalando — equivalente para nuestro uso, depende del stack que ya operes.
- **Sabor de K8s:** Rancher/OpenShift/kubeadm/Talos — agnóstico al diagrama.
- **MinIO vs Ceph:** si ya tenés Ceph, usalo; MinIO es más simple si vas greenfield.
- **Modelo:** Mistral Small 24B es el validado en el lab. En GPU CUDA full-precision podríamos subir a Mistral Large, Llama 3.1 70B (con TP), Qwen 2.5 32B, etc. Depende del trade-off latencia/calidad.
