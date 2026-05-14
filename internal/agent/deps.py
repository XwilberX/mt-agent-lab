"""AgentDeps: estructura que se inyecta como ctx.deps al agent.

Lo crítico: tenant_slug y tenant_name viven acá. El modelo NUNCA los recibe
como parámetro de tool — se cierran desde el contexto del request.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AgentDeps:
    tenant_slug: str
    tenant_name: str
    data_root: Path  # base dir donde viven los parquets del lab
    user_id: str

    @property
    def parquet_dir(self) -> Path:
        """Directorio parquet específico del tenant (read-only en la tool)."""
        return self.data_root / self.tenant_slug
