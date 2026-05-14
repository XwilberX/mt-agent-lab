#!/usr/bin/env python3
"""Benchmark de concurrencia multi-tenant.

Lanza N requests concurrentes contra /t/{slug}/ask mezclando preguntas
de los 3 tenants (acme/beta/gamma) y mide:
  - P50 / P95 / P99 de latencia
  - Throughput (req/s)
  - Errores
  - Validación funcional: las respuestas mencionan el conteo CORRECTO
    de cada tenant (no se cruzan datos bajo carga)

Uso:
    uv run python scripts/bench_concurrency.py --concurrency 1
    uv run python scripts/bench_concurrency.py --concurrency 5
    uv run python scripts/bench_concurrency.py --concurrency 10
    uv run python scripts/bench_concurrency.py --sweep   # corre 1, 5, 10
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from dataclasses import dataclass

import httpx

BASE_URL = "http://localhost:8000"

# Volúmenes esperados por tenant (de los seeds scale=1.0)
EXPECTED_ORDERS = {"acme": 10_000_000, "beta": 1_000_000, "gamma": 50_000}

# Preguntas golden por tenant. Mezcla simples + agregaciones para ejercitar
# la tool consultar. Las respuestas correctas deben mencionar números
# específicos que validamos.
GOLDEN: dict[str, list[tuple[str, str]]] = {
    "acme": [
        ("¿Cuántas órdenes totales tenemos?", "10000000"),
        ("¿Cuántos clientes únicos hay?", None),
        ("¿Cuántas órdenes están en estado 'cancelada'?", None),
        ("¿Cuál es el código postal más frecuente?", None),
    ],
    "beta": [
        ("¿Cuántas órdenes totales tenemos?", "1000000"),
        ("¿Cuántos productos hay en el catálogo?", None),
        ("¿Cuántas reseñas con calificación 5?", None),
        ("¿Cuántas órdenes pagadas con tarjeta_credito?", None),
    ],
    "gamma": [
        ("¿Cuántas órdenes totales tenemos?", "50000"),
        ("¿Cuántos vendedores tenemos?", None),
        ("¿Cuál es el ticket promedio de los pagos?", None),
        ("¿Cuántas órdenes están entregadas?", None),
    ],
}


@dataclass
class Result:
    tenant: str
    question: str
    expected_number: str | None
    status: int
    duration_ms: int
    answer: str
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.status == 200 and self.error is None

    @property
    def answer_correct(self) -> bool | None:
        """Si la pregunta tiene un número esperado, ¿lo menciona la respuesta?
        None si no hay validación posible."""
        if not self.expected_number or not self.ok:
            return None
        norm = self.answer.replace(",", "").replace(".", "").replace(" ", "")
        return self.expected_number in norm


async def fire_one(
    client: httpx.AsyncClient,
    tenant: str,
    question: str,
    expected: str | None,
    user_id: str,
) -> Result:
    t0 = time.monotonic()
    try:
        r = await client.post(
            f"/t/{tenant}/ask",
            json={"question": question},
            headers={"X-User-Id": user_id},
        )
        duration_ms = int((time.monotonic() - t0) * 1000)
        if r.status_code == 200:
            answer = r.json().get("answer", "")
            return Result(tenant, question, expected, 200, duration_ms, answer)
        else:
            return Result(
                tenant, question, expected, r.status_code, duration_ms, "",
                error=f"HTTP {r.status_code}: {r.text[:200]}",
            )
    except Exception as e:
        duration_ms = int((time.monotonic() - t0) * 1000)
        return Result(tenant, question, expected, 0, duration_ms, "", error=repr(e))


def build_task_list(reps: int) -> list[tuple[str, str, str | None]]:
    """Genera la lista de (tenant, question, expected) intercalada por tenant."""
    tasks = []
    for _ in range(reps):
        for tenant, qs in GOLDEN.items():
            for question, expected in qs:
                tasks.append((tenant, question, expected))
    return tasks


async def run_bench(concurrency: int, reps: int) -> list[Result]:
    """Lanza task_list con semáforo de tamaño `concurrency`."""
    tasks_def = build_task_list(reps)
    sem = asyncio.Semaphore(concurrency)
    results: list[Result] = []

    async with httpx.AsyncClient(base_url=BASE_URL, timeout=600) as client:
        async def worker(i: int, tenant: str, q: str, exp: str | None):
            async with sem:
                r = await fire_one(client, tenant, q, exp, f"bench-{i}")
                results.append(r)

        await asyncio.gather(*[
            worker(i, *t) for i, t in enumerate(tasks_def)
        ])

    return results


def summarize(results: list[Result], concurrency: int, wall_seconds: float) -> dict:
    oks = [r for r in results if r.ok]
    fails = [r for r in results if not r.ok]
    durations = sorted(r.duration_ms for r in oks)
    correctness_checks = [r.answer_correct for r in oks if r.answer_correct is not None]
    correct = sum(1 for c in correctness_checks if c)
    n_checks = len(correctness_checks)

    def pct(p: float) -> int:
        if not durations:
            return 0
        k = max(0, min(len(durations) - 1, int(round((p / 100) * (len(durations) - 1)))))
        return durations[k]

    return {
        "concurrency": concurrency,
        "n_total": len(results),
        "n_ok": len(oks),
        "n_fail": len(fails),
        "wall_seconds": round(wall_seconds, 1),
        "throughput_req_per_s": round(len(results) / wall_seconds, 2) if wall_seconds > 0 else 0,
        "p50_ms": pct(50),
        "p95_ms": pct(95),
        "p99_ms": pct(99),
        "min_ms": durations[0] if durations else 0,
        "max_ms": durations[-1] if durations else 0,
        "correctness_pct": round(100 * correct / n_checks, 1) if n_checks else None,
        "correctness_n": f"{correct}/{n_checks}",
    }


def print_report(summaries: list[dict]) -> None:
    print()
    print("=" * 90)
    print(f"{'conc':>5}  {'reqs':>5}  {'ok':>4}  {'fail':>4}  "
          f"{'wall':>6}  {'thru':>7}  {'p50':>6}  {'p95':>6}  {'p99':>6}  {'correct':>10}")
    print("-" * 90)
    for s in summaries:
        print(
            f"{s['concurrency']:>5}  {s['n_total']:>5}  {s['n_ok']:>4}  {s['n_fail']:>4}  "
            f"{s['wall_seconds']:>5}s  {s['throughput_req_per_s']:>6}/s  "
            f"{s['p50_ms']:>5}ms  {s['p95_ms']:>5}ms  {s['p99_ms']:>5}ms  "
            f"{s['correctness_n']:>10}"
        )
    print("=" * 90)


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--concurrency", type=int, default=1)
    ap.add_argument("--reps", type=int, default=1,
                    help="Cantidad de veces que se repite el set golden")
    ap.add_argument("--sweep", action="store_true",
                    help="Corre 1/5/10 concurrencia, ignora --concurrency")
    args = ap.parse_args()

    # smoke test
    async with httpx.AsyncClient(base_url=BASE_URL, timeout=10) as c:
        r = await c.get("/health")
        r.raise_for_status()

    summaries = []
    levels = [1, 5, 10] if args.sweep else [args.concurrency]

    for c in levels:
        n = len(build_task_list(args.reps))
        print(f"\n→ concurrencia={c}  reqs={n}  (esperá: cada request ~5-15s)")
        t0 = time.monotonic()
        results = await run_bench(concurrency=c, reps=args.reps)
        wall = time.monotonic() - t0
        s = summarize(results, c, wall)
        summaries.append(s)
        print(f"  done. wall={s['wall_seconds']}s  p50={s['p50_ms']}ms  "
              f"p95={s['p95_ms']}ms  correct={s['correctness_n']}")
        if s["n_fail"] > 0:
            print("  errores:")
            for r in results:
                if not r.ok:
                    print(f"    {r.tenant} '{r.question[:40]}...' → {r.error}")

    print_report(summaries)

    # Crítico: ¿hubo correctness fails que sugieran cruce de datos?
    for s in summaries:
        if s["correctness_pct"] is not None and s["correctness_pct"] < 100:
            print(f"⚠ Atención: con concurrencia={s['concurrency']}, "
                  f"correctness={s['correctness_pct']}% — revisar respuestas")


if __name__ == "__main__":
    asyncio.run(main())
