#!/usr/bin/env python3
"""Generador híbrido: Olist real como base + multiplicador sintético.

Proceso:
1. Lee los 9 CSVs de Olist desde data/olist/ (correr download_olist.py antes).
2. Traduce columnas y enums clave a español. Ciudades/estados brasileños quedan
   como están (son los datos reales del dataset).
3. Particiona las ~99K órdenes en 3 grupos NO-SUPERPUESTOS (acme/beta/gamma).
   Cada grupo es ~33K órdenes reales con sus customers/products/sellers/items/
   payments/reviews relacionados.
4. Por cada tenant: toma la partición real + genera sintético adicional hasta
   alcanzar el target volume. Los sintéticos referencian las mismas IDs reales
   del tenant (clientes/productos/vendedores recurrentes — patrón realista).
5. Escribe parquet por tabla por tenant.

Uso:
    uv run python scripts/seed_olist.py --scale 1.0    # 10M/1M/50K
    uv run python scripts/seed_olist.py --scale 0.01   # 100K/10K/500 rápido
    uv run python scripts/seed_olist.py --only acme    # solo un tenant
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

ROOT = Path(__file__).resolve().parent.parent
OLIST_DIR = ROOT / "data" / "olist"
OUT_ROOT = ROOT / "data" / "parquet"


# ───────── Mapeos de traducción ─────────

ORDER_STATUS_ES = {
    "approved": "aprobada",
    "canceled": "cancelada",
    "created": "creada",
    "delivered": "entregada",
    "invoiced": "facturada",
    "processing": "en_proceso",
    "shipped": "enviada",
    "unavailable": "no_disponible",
}

PAYMENT_TYPE_ES = {
    "boleto": "boleto",  # producto bancario BR, queda
    "credit_card": "tarjeta_credito",
    "debit_card": "tarjeta_debito",
    "not_defined": "no_definido",
    "voucher": "vales",
}

# 73 categorías Olist (portugués) → español
CATEGORY_PT_ES: dict[str, str] = {
    "agro_industria_e_comercio": "agro_industria",
    "alimentos": "alimentos",
    "alimentos_bebidas": "alimentos_bebidas",
    "artes": "arte",
    "artes_e_artesanato": "arte_y_artesania",
    "artigos_de_festas": "articulos_fiesta",
    "artigos_de_natal": "articulos_navidad",
    "audio": "audio",
    "automotivo": "automotriz",
    "bebes": "bebes",
    "bebidas": "bebidas",
    "beleza_saude": "salud_belleza",
    "brinquedos": "juguetes",
    "cama_mesa_banho": "hogar_textil",
    "casa_conforto": "hogar_confort",
    "casa_conforto_2": "hogar_confort_2",
    "casa_construcao": "construccion",
    "cds_dvds_musicais": "cds_dvds",
    "cine_foto": "fotografia",
    "climatizacao": "climatizacion",
    "comercio": "comercio",
    "construcao_ferramentas_construcao": "construccion_herramientas",
    "construcao_ferramentas_ferramentas": "herramientas",
    "construcao_ferramentas_iluminacao": "iluminacion",
    "construcao_ferramentas_jardim": "jardin",
    "construcao_ferramentas_seguranca": "seguridad",
    "consoles_games": "consolas_videojuegos",
    "cool_stuff": "cool_stuff",
    "dvds_blu_ray": "dvds_blu_ray",
    "eletrodomesticos": "electrodomesticos",
    "eletrodomesticos_2": "electrodomesticos_2",
    "eletronicos": "electronicos",
    "eletroportateis": "pequenos_electrodomesticos",
    "esporte_lazer": "deportes",
    "fashion_bolsas_e_acessorios": "moda_bolsas_accesorios",
    "fashion_calcados": "moda_calzado",
    "fashion_esporte": "moda_deportiva",
    "fashion_infanto_juvenil": "moda_infantil",
    "fashion_roupa_feminina": "moda_femenina",
    "fashion_roupa_infanto_juvenil": "moda_infantil_juvenil",
    "fashion_roupa_masculina": "moda_masculina",
    "fashion_underwear_e_moda_praia": "moda_intima_playa",
    "ferramentas_jardim": "jardin_herramientas",
    "flores": "flores",
    "fraldas_higiene": "panales_higiene",
    "industria_comercio_e_negocios": "industria_negocios",
    "informatica_acessorios": "computacion",
    "instrumentos_musicais": "instrumentos_musicales",
    "la_cuisine": "cocina_gourmet",
    "livros_importados": "libros_importados",
    "livros_interesse_geral": "libros",
    "livros_tecnicos": "libros_tecnicos",
    "malas_acessorios": "maletas_accesorios",
    "market_place": "marketplace",
    "moveis_colchao_e_estofado": "muebles_colchones",
    "moveis_cozinha_area_de_servico_jantar_e_jardim": "muebles_cocina",
    "moveis_decoracao": "muebles_decoracion",
    "moveis_escritorio": "muebles_oficina",
    "moveis_quarto": "muebles_recamara",
    "moveis_sala": "muebles_sala",
    "musica": "musica",
    "papelaria": "papeleria",
    "pcs": "computadoras",
    "pc_gamer": "pc_gamer",
    "perfumaria": "perfumeria",
    "pet_shop": "mascotas",
    "portateis_casa_forno_e_cafe": "portatiles_hogar",
    "portateis_cozinha_e_preparadores_de_alimentos": "portatiles_cocina",
    "relogios_presentes": "relojes_regalos",
    "seguros_e_servicos": "seguros_servicios",
    "sinalizacao_e_seguranca": "senalizacion_seguridad",
    "tablets_impressao_imagem": "tablets_impresion",
    "telefonia": "telefonia",
    "telefonia_fixa": "telefonia_fija",
    "utilidades_domesticas": "hogar_utiles",
}

# ───────── Perfiles ─────────

@dataclass(frozen=True)
class TenantProfile:
    slug: str
    name: str
    target_orders: int  # total final (real + sintético)
    seed: int


def make_profiles(scale: float) -> list[TenantProfile]:
    def s(n: int) -> int:
        return max(1, int(n * scale))

    return [
        TenantProfile("acme", "Tienda Acme", s(10_000_000), seed=42),
        TenantProfile("beta", "Beta Electronics", s(1_000_000), seed=43),
        TenantProfile("gamma", "Gamma Nueva", s(50_000), seed=44),
    ]


# ───────── Carga + traducción de Olist ─────────

def load_olist() -> dict[str, pl.DataFrame]:
    """Carga los 9 CSVs y los traduce a esquema español."""
    print("Cargando Olist...")

    # Translation file (no usado directamente acá pero queda accesible)
    cats = pl.read_csv(OLIST_DIR / "product_category_name_translation.csv")

    orders = pl.read_csv(
        OLIST_DIR / "olist_orders_dataset.csv",
        try_parse_dates=True,
    ).rename({
        "order_id": "orden_id",
        "customer_id": "cliente_id",
        "order_status": "estado",
        "order_purchase_timestamp": "fecha_compra",
        "order_approved_at": "fecha_aprobacion",
        "order_delivered_carrier_date": "fecha_envio",
        "order_delivered_customer_date": "fecha_entrega_real",
        "order_estimated_delivery_date": "fecha_entrega_estimada",
    }).with_columns(
        pl.col("estado").replace_strict(ORDER_STATUS_ES, default=pl.col("estado")),
    )

    customers = pl.read_csv(OLIST_DIR / "olist_customers_dataset.csv").rename({
        "customer_id": "cliente_id",
        "customer_unique_id": "cliente_unico_id",
        "customer_zip_code_prefix": "codigo_postal",
        "customer_city": "ciudad",
        "customer_state": "estado_codigo",
    })

    sellers = pl.read_csv(OLIST_DIR / "olist_sellers_dataset.csv").rename({
        "seller_id": "vendedor_id",
        "seller_zip_code_prefix": "codigo_postal",
        "seller_city": "ciudad",
        "seller_state": "estado_codigo",
    })

    items = pl.read_csv(
        OLIST_DIR / "olist_order_items_dataset.csv",
        try_parse_dates=True,
    ).rename({
        "order_id": "orden_id",
        "order_item_id": "item_seq",
        "product_id": "producto_id",
        "seller_id": "vendedor_id",
        "shipping_limit_date": "fecha_limite_envio",
        "price": "precio",
        "freight_value": "flete",
    })

    payments = pl.read_csv(OLIST_DIR / "olist_order_payments_dataset.csv").rename({
        "order_id": "orden_id",
        "payment_sequential": "secuencia",
        "payment_type": "tipo_pago",
        "payment_installments": "mensualidades",
        "payment_value": "valor",
    }).with_columns(
        pl.col("tipo_pago").replace_strict(PAYMENT_TYPE_ES, default=pl.col("tipo_pago")),
    )

    reviews = pl.read_csv(
        OLIST_DIR / "olist_order_reviews_dataset.csv",
        try_parse_dates=True,
    ).rename({
        "review_id": "resena_id",
        "order_id": "orden_id",
        "review_score": "calificacion",
        "review_comment_title": "titulo",
        "review_comment_message": "comentario",
        "review_creation_date": "fecha_creacion",
        "review_answer_timestamp": "fecha_respuesta",
    })

    products = pl.read_csv(OLIST_DIR / "olist_products_dataset.csv").rename({
        "product_id": "producto_id",
        "product_category_name": "categoria_pt",
        "product_name_lenght": "nombre_caracteres",
        "product_description_lenght": "descripcion_caracteres",
        "product_photos_qty": "fotos_cantidad",
        "product_weight_g": "peso_gramos",
        "product_length_cm": "largo_cm",
        "product_height_cm": "alto_cm",
        "product_width_cm": "ancho_cm",
    }).with_columns(
        pl.col("categoria_pt")
            .replace_strict(CATEGORY_PT_ES, default=pl.lit("otros"))
            .alias("categoria"),
    ).drop("categoria_pt")

    # Catálogo de categorías único
    categorias = pl.DataFrame({
        "categoria": sorted(set(CATEGORY_PT_ES.values()) | {"otros"}),
    })

    print(f"  orders: {len(orders):,}  customers: {len(customers):,}  "
          f"items: {len(items):,}  products: {len(products):,}  "
          f"sellers: {len(sellers):,}  payments: {len(payments):,}  "
          f"reviews: {len(reviews):,}")

    return {
        "orders": orders,
        "customers": customers,
        "items": items,
        "payments": payments,
        "reviews": reviews,
        "products": products,
        "sellers": sellers,
        "categorias": categorias,
    }


# ───────── Partición no-superpuesta ─────────

def partition_real(olist: dict[str, pl.DataFrame], n_tenants: int = 3, seed: int = 0
                   ) -> list[dict[str, pl.DataFrame]]:
    """Particiona las órdenes en N grupos disjuntos y arrastra las filas relacionadas."""
    orders = olist["orders"]
    n = len(orders)

    # Shuffle reproducible
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)

    chunks = np.array_split(idx, n_tenants)
    result = []
    for chunk_idx in chunks:
        # filas de orders en este chunk
        ords = orders[chunk_idx.tolist()]

        # joins de membership (rápido y sin warnings de is_in)
        order_keys = ords.select("orden_id")
        client_keys = ords.select("cliente_id").unique()
        its = olist["items"].join(order_keys, on="orden_id", how="inner")
        prod_keys = its.select("producto_id").unique()
        vend_keys = its.select("vendedor_id").unique()

        cus = olist["customers"].join(client_keys, on="cliente_id", how="inner")
        pays = olist["payments"].join(order_keys, on="orden_id", how="inner")
        revs = olist["reviews"].join(order_keys, on="orden_id", how="inner")
        prods = olist["products"].join(prod_keys, on="producto_id", how="inner")
        vens = olist["sellers"].join(vend_keys, on="vendedor_id", how="inner")

        result.append({
            "orders": ords,
            "customers": cus,
            "items": its,
            "payments": pays,
            "reviews": revs,
            "products": prods,
            "sellers": vens,
            "categorias": olist["categorias"],
        })
    return result


# ───────── Multiplicador sintético ─────────

def multiplicar_sintetico(
    real: dict[str, pl.DataFrame],
    slug: str,
    target_orders: int,
    seed: int,
) -> dict[str, pl.DataFrame]:
    """Genera órdenes sintéticas adicionales hasta `target_orders` total.

    Las nuevas órdenes referencian los mismos customers/products/sellers del
    pool real (recurrentes). IDs sintéticos llevan prefijo 's' tras el slug
    inicial, para no colisionar con los UUIDs Olist.
    """
    real_orders = real["orders"]
    n_real = len(real_orders)
    n_syn = max(0, target_orders - n_real)

    out = {k: v.clone() for k, v in real.items()}
    if n_syn == 0:
        return out

    rng = np.random.default_rng(seed)

    # Pools
    cliente_pool = real["customers"]["cliente_id"].to_numpy()
    prod_pool = real["products"]["producto_id"].to_numpy()
    vend_pool = real["sellers"]["vendedor_id"].to_numpy()

    # Stats reales para imitar distribuciones
    precios_reales = real["items"]["precio"].to_numpy()
    fletes_reales = real["items"]["flete"].to_numpy()
    log_precio_mu = float(np.log(np.maximum(precios_reales, 1.0)).mean())
    log_precio_sigma = float(np.log(np.maximum(precios_reales, 1.0)).std())

    fecha_min = real_orders["fecha_compra"].min()
    fecha_max = real_orders["fecha_compra"].max()

    # IDs sintéticos: prefijo `s{slug[0]}o` para no chocar con UUIDs reales
    pfx = f"s{slug[0]}o"
    syn_order_ids = np.array([f"{pfx}{i:09x}" for i in range(n_syn)])

    # Clientes: sample con reemplazo (clientes recurrentes)
    syn_cliente_ids = cliente_pool[rng.integers(0, len(cliente_pool), size=n_syn)]

    # Estados: tomar la distribución real
    estados_reales, counts = np.unique(real_orders["estado"].to_numpy(), return_counts=True)
    estado_probs = counts / counts.sum()
    syn_estados = rng.choice(estados_reales, size=n_syn, p=estado_probs)

    # Fechas: uniformes en el rango real, extendidas hasta 2025-12-31 para acme/beta
    fecha_min_ms = np.datetime64(fecha_min, "ms")
    fecha_max_extendida = np.datetime64("2025-12-31", "ms")
    span_ms = int((fecha_max_extendida - fecha_min_ms).astype(np.int64))
    syn_fecha_compra = fecha_min_ms + rng.integers(0, span_ms, size=n_syn).astype("timedelta64[ms]")

    aprobacion_offset = (rng.integers(300, 7200, size=n_syn) * 1000).astype("timedelta64[ms]")
    entrega_est_offset = (rng.integers(5, 14, size=n_syn) * 86_400_000).astype("timedelta64[ms]")
    entrega_real_offset = (rng.integers(3, 21, size=n_syn) * 86_400_000).astype("timedelta64[ms]")

    syn_aprob = syn_fecha_compra + aprobacion_offset
    syn_envio = syn_fecha_compra + (rng.integers(1, 5, size=n_syn) * 86_400_000).astype("timedelta64[ms]")
    syn_entrega_est = syn_fecha_compra + entrega_est_offset
    syn_entrega_real = syn_fecha_compra + entrega_real_offset

    nat = np.datetime64("NaT", "ms")
    aprob_mask = syn_estados != "cancelada"
    entregada_mask = syn_estados == "entregada"
    envio_mask = (syn_estados == "enviada") | entregada_mask

    syn_ords_df = pl.DataFrame({
        "orden_id": syn_order_ids,
        "cliente_id": syn_cliente_ids,
        "estado": syn_estados,
        "fecha_compra": syn_fecha_compra,
        "fecha_aprobacion": np.where(aprob_mask, syn_aprob, nat),
        "fecha_envio": np.where(envio_mask, syn_envio, nat),
        "fecha_entrega_real": np.where(entregada_mask, syn_entrega_real, nat),
        "fecha_entrega_estimada": syn_entrega_est,
    })

    # Alineación de schema con orders reales
    syn_ords_df = syn_ords_df.select(real_orders.columns)
    out["orders"] = pl.concat([real_orders, syn_ords_df], how="vertical_relaxed")

    # === Items: 1-3 items por orden ===
    item_counts = rng.choice([1, 2, 3], size=n_syn, p=[0.85, 0.12, 0.03])
    total_items = int(item_counts.sum())
    orden_idx_rep = np.repeat(np.arange(n_syn), item_counts)
    item_seq = np.concatenate([np.arange(1, k + 1, dtype=np.int64) for k in item_counts])

    syn_prod_idx = rng.integers(0, len(prod_pool), size=total_items)
    syn_vend_idx = rng.integers(0, len(vend_pool), size=total_items)

    # Precios siguiendo lognormal de los reales
    syn_precio = np.round(np.exp(rng.normal(log_precio_mu, log_precio_sigma, size=total_items)), 2)
    syn_precio = np.clip(syn_precio, 5.0, 50_000.0)
    syn_flete = np.round(np.minimum(syn_precio * 0.08, 200.0) + rng.uniform(10, 50, size=total_items), 2)

    # fecha_limite_envio: ~3 días después de la fecha_compra
    syn_fecha_limite = syn_fecha_compra[orden_idx_rep] + (3 * 86_400_000)

    syn_items_df = pl.DataFrame({
        "orden_id": syn_order_ids[orden_idx_rep],
        "item_seq": item_seq,
        "producto_id": prod_pool[syn_prod_idx],
        "vendedor_id": vend_pool[syn_vend_idx],
        "fecha_limite_envio": syn_fecha_limite,
        "precio": syn_precio,
        "flete": syn_flete,
    }).select(real["items"].columns)
    out["items"] = pl.concat([real["items"], syn_items_df], how="vertical_relaxed")

    # === Pagos: 1 por orden ===
    payment_types, counts = np.unique(real["payments"]["tipo_pago"].to_numpy(), return_counts=True)
    pt_probs = counts / counts.sum()
    syn_tp = rng.choice(payment_types, size=n_syn, p=pt_probs)
    # mensualidades: solo si tarjeta_credito
    mens_credito = rng.choice([1, 2, 3, 4, 5, 6, 10, 12], size=n_syn,
                              p=[0.30, 0.10, 0.15, 0.10, 0.10, 0.15, 0.05, 0.05])
    syn_mens = np.where(syn_tp == "tarjeta_credito", mens_credito, 1).astype(np.int64)

    # valor: sum items por orden
    syn_valor_por_orden = (
        syn_items_df.group_by("orden_id")
        .agg((pl.col("precio") + pl.col("flete")).sum().alias("valor"))
    )
    ord_df = pl.DataFrame({"orden_id": syn_order_ids, "_seq": np.arange(n_syn, dtype=np.int64)})
    joined = ord_df.join(syn_valor_por_orden, on="orden_id", how="left").sort("_seq")
    syn_valor = joined["valor"].fill_null(0.0).to_numpy()

    syn_pays_df = pl.DataFrame({
        "orden_id": syn_order_ids,
        "secuencia": np.ones(n_syn, dtype=np.int64),
        "tipo_pago": syn_tp,
        "mensualidades": syn_mens,
        "valor": np.round(syn_valor, 2),
    }).select(real["payments"].columns)
    out["payments"] = pl.concat([real["payments"], syn_pays_df], how="vertical_relaxed")

    # === Reseñas: ~60% de entregadas ===
    review_mask = entregada_mask & (rng.random(n_syn) < 0.6)
    n_reviews = int(review_mask.sum())
    review_ords = syn_order_ids[review_mask]
    review_ids = np.array([f"s{slug[0]}r{i:09x}" for i in range(n_reviews)])
    calif = rng.choice([5, 4, 3, 2, 1], size=n_reviews,
                       p=[0.57, 0.20, 0.08, 0.05, 0.10]).astype(np.int64)
    fecha_creacion = syn_entrega_real[review_mask] + (rng.integers(0, 7, size=n_reviews) * 86_400_000).astype("timedelta64[ms]")
    fecha_respuesta = fecha_creacion + (rng.integers(1, 5, size=n_reviews) * 86_400_000).astype("timedelta64[ms]")

    syn_revs_df = pl.DataFrame({
        "resena_id": review_ids,
        "orden_id": review_ords,
        "calificacion": calif,
        "titulo": pl.Series([None] * n_reviews, dtype=pl.Utf8),
        "comentario": pl.Series([None] * n_reviews, dtype=pl.Utf8),
        "fecha_creacion": fecha_creacion,
        "fecha_respuesta": fecha_respuesta,
    }).select(real["reviews"].columns)
    out["reviews"] = pl.concat([real["reviews"], syn_revs_df], how="vertical_relaxed")

    return out


# ───────── Driver ─────────

def seed_tenant(p: TenantProfile, partition: dict[str, pl.DataFrame]) -> None:
    t0 = time.time()
    n_real_full = len(partition["orders"])

    # Si target < real disponible, truncar para no inflar (típico en --scale chico)
    if p.target_orders < n_real_full:
        truncated_orders = partition["orders"].head(p.target_orders)
        order_keys = truncated_orders.select("orden_id")
        client_keys = truncated_orders.select("cliente_id").unique()
        partition = {
            **partition,
            "orders": truncated_orders,
            "customers": partition["customers"].join(client_keys, on="cliente_id", how="inner"),
            "items": partition["items"].join(order_keys, on="orden_id", how="inner"),
            "payments": partition["payments"].join(order_keys, on="orden_id", how="inner"),
            "reviews": partition["reviews"].join(order_keys, on="orden_id", how="inner"),
        }

    n_real = len(partition["orders"])
    n_syn = max(0, p.target_orders - n_real)
    print(f"\n=== {p.slug} ({p.name}) ===")
    print(f"  Target: {p.target_orders:,}  Real: {n_real:,}  Sintético: {n_syn:,}")

    final = multiplicar_sintetico(partition, p.slug, p.target_orders, p.seed)

    out_dir = OUT_ROOT / p.slug
    out_dir.mkdir(parents=True, exist_ok=True)

    tablas = ["orders", "customers", "items", "payments", "reviews", "products", "sellers", "categorias"]
    nombres_es = {
        "orders": "ordenes", "customers": "clientes", "items": "items",
        "payments": "pagos", "reviews": "resenas", "products": "productos",
        "sellers": "vendedores", "categorias": "categorias",
    }
    for t in tablas:
        path = out_dir / f"{nombres_es[t]}.parquet"
        final[t].write_parquet(path)
        print(f"  {nombres_es[t]:12} {len(final[t]):>12,}   t={time.time()-t0:5.1f}s")

    print(f"  ───────────────────────────────────────────")
    print(f"  Total: {time.time()-t0:.1f}s")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", type=float, default=1.0,
                    help="Multiplicador de tamaños (1.0=README sizes)")
    ap.add_argument("--only", type=str, default=None,
                    help="Generar solo este tenant (acme/beta/gamma)")
    args = ap.parse_args()

    if not (OLIST_DIR / "olist_orders_dataset.csv").exists():
        print("ERROR: data/olist/ vacío. Corré primero: uv run python scripts/download_olist.py")
        raise SystemExit(1)

    olist = load_olist()
    partitions = partition_real(olist, n_tenants=3, seed=2026)
    profiles = make_profiles(args.scale)

    for p, part in zip(profiles, partitions):
        if args.only and p.slug != args.only:
            continue
        seed_tenant(p, part)

    print("\n✅ Seed completado")


if __name__ == "__main__":
    main()
