"""
Aero-Grid Wind AI: Master Dataset Merging Pipeline
===================================================
Merges:
  1. AI Member 1: Atmospheric reanalysis weather dataset (site_weather_summary.csv)
  2. AI Member 2: Spatial & GIS terrain dataset (site_spatial_summary.csv)
Produces:
  -> data/site_master_dataset.csv
"""

import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

WEATHER_CSV = os.path.join(DATA_DIR, "site_weather_summary.csv")
SPATIAL_CSV = os.path.join(DATA_DIR, "site_spatial_summary.csv")
MASTER_CSV = os.path.join(DATA_DIR, "site_master_dataset.csv")


def merge_datasets() -> pd.DataFrame:
    print("=" * 85)
    print("AERO-GRID: Master Dataset Integration (AI #1 Weather + AI #2 Spatial/GIS)")
    print("=" * 85)

    if not os.path.exists(WEATHER_CSV):
        raise FileNotFoundError(f"Weather dataset missing at: {WEATHER_CSV}")
    if not os.path.exists(SPATIAL_CSV):
        raise FileNotFoundError(f"Spatial dataset missing at: {SPATIAL_CSV}")

    df_weather = pd.read_csv(WEATHER_CSV)
    df_spatial = pd.read_csv(SPATIAL_CSV)

    print(f"[*] Loaded AI Member 1 Weather Dataset: {df_weather.shape[0]} sites, {df_weather.shape[1]} columns")
    print(f"[*] Loaded AI Member 2 Spatial Dataset: {df_spatial.shape[0]} sites, {df_spatial.shape[1]} columns")

    # Spatial GIS columns to append to weather data
    spatial_cols_to_add = [
        "site_id",
        "elevation_m",
        "slope_degrees",
        "aspect_degrees",
        "slope_category",
        "land_cover_category",
        "roughness_class",
        "roughness_length_z0_m",
        "osm_village",
        "osm_county",
        "osm_district",
    ]

    df_spatial_subset = df_spatial[spatial_cols_to_add]

    # Perform inner merge on primary key site_id
    df_master = pd.merge(df_weather, df_spatial_subset, on="site_id", how="inner")

    # Verification checks
    if len(df_master) != len(df_weather):
        raise ValueError(f"Row count mismatch after merge! Expected {len(df_weather)}, got {len(df_master)}")

    null_counts = df_master.isnull().sum()
    if null_counts.any():
        print(f"[!] Warning: Missing values detected:\n{null_counts[null_counts > 0]}")
    else:
        print("[+] Data Integrity Check: Passed (0 missing values across all columns)")

    # Reorder columns logically for downstream AI Member 3 & 4
    column_order = [
        # 1. Geographic & Site Identity
        "site_id",
        "site_name",
        "category",
        "latitude",
        "longitude",
        "osm_village",
        "osm_county",
        "osm_district",
        
        # 2. AI Member 2: Spatial & Topographic Metrics
        "elevation_m",
        "slope_degrees",
        "aspect_degrees",
        "slope_category",
        "land_cover_category",
        "roughness_class",
        "roughness_length_z0_m",

        # 3. AI Member 1: Wind & Atmospheric Reanalysis Metrics
        "avg_wind_speed_10m",
        "max_wind_speed_10m",
        "p90_wind_speed_10m",
        "avg_gust_speed_10m",
        "avg_wpd_w_m2",
        "operable_hours_pct",
        "avg_temp_c",
        "avg_air_density_kg_m3",
        "prevailing_wind_dir_deg",
        "prevailing_wind_dir_cardinal",

        # 4. Metadata
        "total_recorded_hours",
        "time_window"
    ]

    # Ensure all columns exist
    df_master = df_master[column_order]

    # Save master dataset
    df_master.to_csv(MASTER_CSV, index=False)
    print(f"\n[+] SUCCESS: Master Dataset exported successfully to:")
    print(f"    -> {MASTER_CSV}")
    print(f"    Total Sites: {len(df_master)} | Total Features: {df_master.shape[1]}")

    return df_master


if __name__ == "__main__":
    df_master = merge_datasets()
    print("\n" + "=" * 85)
    print("MASTER DATASET PREVIEW (Ready for AI #3 ML Scoring & AI #4 Interactive Mapping):")
    print("=" * 85)
    preview_cols = [
        "site_id", "site_name", "elevation_m", "slope_degrees",
        "roughness_length_z0_m", "avg_wind_speed_10m", "avg_wpd_w_m2"
    ]
    print(df_master[preview_cols].to_string(index=False))
    print("=" * 85)
