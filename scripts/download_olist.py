#!/usr/bin/env python3
"""Baja los 9 CSVs raw de Olist desde el mirror oficial en GitHub.

Sin auth, idempotente: si los archivos ya existen y no están vacíos, los
saltea. Para forzar redescarga: borrar data/olist/*.csv.

Uso:
    uv run python scripts/download_olist.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import httpx

BASE = "https://raw.githubusercontent.com/olist/work-at-olist-data/master/datasets"

FILES = [
    "olist_customers_dataset.csv",
    "olist_geolocation_dataset.csv",
    "olist_order_items_dataset.csv",
    "olist_order_payments_dataset.csv",
    "olist_order_reviews_dataset.csv",
    "olist_orders_dataset.csv",
    "olist_products_dataset.csv",
    "olist_sellers_dataset.csv",
    "product_category_name_translation.csv",
]

OUT = Path(__file__).resolve().parent.parent / "data" / "olist"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    total_bytes = 0
    t0 = time.time()
    with httpx.Client(timeout=120, follow_redirects=True) as client:
        for f in FILES:
            dest = OUT / f
            if dest.exists() and dest.stat().st_size > 0:
                print(f"  ✓ {f:50}  cached ({dest.stat().st_size:>12,} B)")
                continue
            url = f"{BASE}/{f}"
            print(f"  ↓ {f:50}  ", end="", flush=True)
            try:
                r = client.get(url)
                r.raise_for_status()
            except httpx.HTTPError as e:
                print(f"\n  ERROR bajando {url}: {e}", file=sys.stderr)
                sys.exit(1)
            dest.write_bytes(r.content)
            total_bytes += len(r.content)
            print(f"{len(r.content):>12,} B")

    print(f"\n✅ Olist descargado ({total_bytes:,} B en {time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
