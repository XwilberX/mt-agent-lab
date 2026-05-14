-- Inicialización del Postgres del lab.
-- Tabla de audit en `public` (global), una por tenant en su propio schema.

CREATE EXTENSION IF NOT EXISTS vector;

-- Catálogo de tenants registrados en el lab
CREATE TABLE IF NOT EXISTS public.tenants (
    slug TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    schema_name TEXT NOT NULL UNIQUE,
    enabled BOOLEAN NOT NULL DEFAULT TRUE,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Audit log global: cada request al agente queda aquí
CREATE TABLE IF NOT EXISTS public.agent_audit (
    id BIGSERIAL PRIMARY KEY,
    tenant_slug TEXT NOT NULL,
    user_id TEXT NOT NULL,
    session_id TEXT,
    prompt TEXT NOT NULL,
    response TEXT,
    tool_calls JSONB NOT NULL DEFAULT '[]'::jsonb,
    duration_ms INTEGER,
    flagged_cross_tenant BOOLEAN NOT NULL DEFAULT FALSE,
    error TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_audit_tenant
    ON public.agent_audit(tenant_slug, created_at DESC);

CREATE INDEX IF NOT EXISTS idx_audit_flagged
    ON public.agent_audit(flagged_cross_tenant)
    WHERE flagged_cross_tenant = TRUE;
