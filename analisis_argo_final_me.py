# analisis_argo_final.py

import os
import logging
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px


# =========================
# CONSTANTES
# =========================
INPUT_FILE = "data/argo_mediterraneo.csv"
OUTPUT_STATS = "data/estadisticas_argo.csv"

HEATMAP_PNG = "outputs/heatmap_temp_mensual.png"
TS_PNG = "outputs/diagram_ts.png"
MAP_HTML = "outputs/mapa_boyas.html"


# =========================
# LOGGING
# =========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


# =========================
# FUNCIONES
# =========================

def cargar_datos(ruta: str) -> pd.DataFrame:
    """
    Carga el dataset ARGO asegurando que la columna 'date' es datetime.
    """
    df = pd.read_csv(ruta, parse_dates=["date"])
    logging.info(f"Datos cargados: {df.shape}")
    return df


def limpiar_datos(df: pd.DataFrame) -> pd.DataFrame:
    """
    Elimina filas con NaN en temperature, salinity o depth.
    """
    df_clean = df.dropna(subset=["temperature", "salinity", "depth"])
    logging.info(f"Datos tras limpieza: {df_clean.shape}")
    return df_clean


def crear_estadisticas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula estadísticas de temperatura y salinidad por cuenca.
    """
    stats = (
        df.groupby("basin")[["temperature", "salinity"]]
        .agg(["mean", "std", "min", "max"])
    )

    logging.info("Estadísticas calculadas por cuenca")
    return stats


def guardar_estadisticas(stats: pd.DataFrame, ruta: str):
    """
    Guarda estadísticas en CSV.
    """
    stats.to_csv(ruta)
    logging.info(f"Estadísticas guardadas en {ruta}")


def heatmap_mensual(df: pd.DataFrame, output_path: str):
    """
    Genera heatmap de temperatura media mensual por cuenca.
    """
    df["month"] = df["date"].dt.month

    pivot = df.pivot_table(
        index="basin",
        columns="month",
        values="temperature",
        aggfunc="mean"
    )

    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot, cmap="coolwarm")

    plt.title("Temperatura media mensual por cuenca")
    plt.xlabel("Mes")
    plt.ylabel("Cuenca")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    logging.info(f"Heatmap guardado en {output_path}")


def diagrama_ts(df: pd.DataFrame, output_path: str):
    """
    Genera diagrama T-S (Temperatura vs Salinidad) por cuenca.
    """
    plt.figure(figsize=(8, 6))

    for basin, group in df.groupby("basin"):
        plt.scatter(
            group["salinity"],
            group["temperature"],
            label=basin,
            alpha=0.5
        )

    plt.xlabel("Salinidad (PSU)")
    plt.ylabel("Temperatura (°C)")
    plt.title("Diagrama T-S por cuenca")
    plt.legend()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    logging.info(f"Diagrama T-S guardado en {output_path}")


def mapa_boyas(df: pd.DataFrame, output_path: str):
    """
    Genera mapa interactivo de distribución de boyas.
    """
    df_map = df.groupby("float_id", as_index=False).agg({
        "latitude": "mean",
        "longitude": "mean",
        "basin": "first"
    })

    fig = px.scatter_geo(
        df_map,
        lat="latitude",
        lon="longitude",
        color="basin",
        projection="natural earth",
        title="Distribución de boyas en el Mediterráneo"
    )

    fig.update_geos(
        center={"lat": 37.5, "lon": 15},
        lataxis_range=[30, 46],
        lonaxis_range=[-6, 36],
        showland=True,
        landcolor="lightgray"
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    fig.write_html(output_path)

    logging.info(f"Mapa guardado en {output_path}")


# =========================
# MAIN
# =========================

def main():
    """
    Pipeline principal de análisis ARGO.
    """

    df = cargar_datos(INPUT_FILE)

    df = limpiar_datos(df)

    stats = crear_estadisticas(df)
    guardar_estadisticas(stats, OUTPUT_STATS)

    heatmap_mensual(df, HEATMAP_PNG)

    diagrama_ts(df, TS_PNG)

    mapa_boyas(df, MAP_HTML)

    logging.info("Pipeline completado correctamente")


if __name__ == "__main__":
    main()