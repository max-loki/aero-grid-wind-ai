import os
import time
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import requests

# 1. CANDIDATE SITING COORDINATES (VIT BHOPAL, ASHTA, KOTHRI, SEHORE RURAL BELT)
# Siting candidates covering agricultural plains, open plateaus, and village outskirts
CANDIDATE_SITES: List[Dict] = [
    {
        "site_id": "AERO_01",
        "name": "VIT Bhopal Campus (Open Ridge)",
        "category": "Educational/Open Ridge",
        "lat": 23.0775,
        "lon": 76.8491,
    },
    {
        "site_id": "AERO_02",
        "name": "Kothrikalan Rural Center",
        "category": "Village Settlement",
        "lat": 23.0712,
        "lon": 76.8420,
    },
    {
        "site_id": "AERO_03",
        "name": "Kothri Farmland North (Parbati Basin)",
        "category": "Agricultural Flatland",
        "lat": 23.0880,
        "lon": 76.8520,
    },
    {
        "site_id": "AERO_04",
        "name": "Metwada Rural Outskirts",
        "category": "Rural Community",
        "lat": 23.0645,
        "lon": 76.8925,
    },
    {
        "site_id": "AERO_05",
        "name": "Pagaria Hat Agricultural Zone",
        "category": "Agricultural Belt",
        "lat": 23.1120,
        "lon": 76.8250,
    },
    {
        "site_id": "AERO_06",
        "name": "Lasudiya Khas (Midway Ashta-Kothri)",
        "category": "Farmland Plateau",
        "lat": 23.0450,
        "lon": 76.7820,
    },
    {
        "site_id": "AERO_07",
        "name": "Ashta Rural East (Farmland Boundary)",
        "category": "Suburban/Rural Border",
        "lat": 23.0210,
        "lon": 76.7320,
    },
    {
        "site_id": "AERO_08",
        "name": "Ashta South Agricultural Plain",
        "category": "Agricultural Flatland",
        "lat": 22.9850,
        "lon": 76.7110,
    },
    {
        "site_id": "AERO_09",
        "name": "Barkheda Hasan Rural Sector",
        "category": "Rural Settlement",
        "lat": 23.1050,
        "lon": 76.8850,
    },
    {
        "site_id": "AERO_10",
        "name": "Ichhawar Rural Plateau (High Elevation)",
        "category": "Hilly Rural Area",
        "lat": 23.0240,
        "lon": 77.0150,
    },
    {
        "site_id": "AERO_11",
        "name": "Sehore Rural South-West Outskirts",
        "category": "Agricultural Cluster",
        "lat": 23.1650,
        "lon": 77.0120,
    },
    {
        "site_id": "AERO_12",
        "name": "Shyampur Rural Outskirts",
        "category": "Village Cluster",
        "lat": 23.2450,
        "lon": 77.0850,
    },
]


# 2. ANALYSIS PARAMETERS & TURBINE SPECIFICATIONS
# Historical date range (2 full calendar years for seasonal fidelity)
START_DATE = "2023-01-01"
END_DATE = "2024-12-31"

# Micro-wind turbine specifications (Standard 1kW - 5kW mast turbine)
CUT_IN_SPEED_MS = 3.0    # Minimum wind speed required to start blade rotation (m/s)
CUT_OUT_SPEED_MS = 25.0  # Maximum safe operating speed before braking/furling (m/s)
R_SPECIFIC_AIR = 287.05  # Specific gas constant for dry air in J / (kg * K)

# Output directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)



# 3. ATMOSPHERIC PHYSICS FUNCTIONS

def calculate_air_density(pressure_hpa: pd.Series, temp_celsius: pd.Series) -> pd.Series:
    """
    Computes local dynamic air density (rho) in kg/m^3 using the Ideal Gas Law:
    rho = P / (R_specific * T)
    where:
      P = pressure in Pascals (hPa * 100)
      T = temperature in Kelvin (°C + 273.15)
      R_specific = 287.05 J/(kg·K)
    """
    pressure_pa = pressure_hpa * 100.0
    temp_kelvin = temp_celsius + 273.15
    return pressure_pa / (R_SPECIFIC_AIR * temp_kelvin)


def vector_mean_wind_direction(directions_deg: pd.Series) -> float:
    """
    Computes the true circular mean of wind angles using vector decomposition.
    Avoids arithmetic average artifacts (e.g., mean of 350° and 10° is 0°/360°, not 180°).
    """
    clean_dirs = directions_deg.dropna()
    if clean_dirs.empty:
        return 0.0
    rad = np.radians(clean_dirs)
    sin_sum = np.sum(np.sin(rad))
    cos_sum = np.sum(np.cos(rad))
    mean_rad = np.arctan2(sin_sum, cos_sum)
    mean_deg = np.degrees(mean_rad)
    return round(float((mean_deg + 360) % 360), 2)


def get_compass_bearing(deg: float) -> str:
    """Converts compass degrees (0-360) into a standard 16-point cardinal compass rose label."""
    cardinals = [
        "N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
        "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"
    ]
    idx = int((deg + 11.25) / 22.5) % 16
    return cardinals[idx]


# 4. DATA EXTRACTION ENGINE (OPEN-METEO HISTORICAL ARCHIVE API)

def fetch_open_meteo_archive(lat: float, lon: float, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Pulls hourly ERA5 reanalysis data at 10m mast height from Open-Meteo.
    API is free, public, and requires no API key.
    """
    endpoint = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": [
            "wind_speed_10m",
            "wind_direction_10m",
            "wind_gusts_10m",
            "temperature_2m",
            "surface_pressure"
        ],
        "wind_speed_unit": "ms",   # Standardized to m/s
        "timezone": "UTC"
    }

    response = requests.get(endpoint, params=params, timeout=35)
    response.raise_for_status()
    data = response.json()

    if "hourly" not in data:
        raise ValueError(f"Open-Meteo returned no hourly payload for coordinates ({lat}, {lon})")

    df = pd.DataFrame(data["hourly"])
    df["time"] = pd.to_datetime(df["time"])
    return df


# 5. DATA PROCESSING & FEATURE ENGINEERING

def process_site(site_meta: Dict, start_date: str, end_date: str) -> Tuple[Dict, pd.DataFrame]:
    """
    Fetches raw timeseries, handles missing data, computes physics metrics,
    and returns both site summary statistics and clean hourly data.
    """
    site_id = site_meta["site_id"]
    lat, lon = site_meta["lat"], site_meta["lon"]
    name = site_meta["name"]
    category = site_meta["category"]

    print(f"[*] Processing {site_id} | {name} (Lat: {lat}, Lon: {lon})...")

    # Fetch raw hourly dataset
    raw_df = fetch_open_meteo_archive(lat, lon, start_date, end_date)
    raw_df["site_id"] = site_id

    # 1. Clean missing values on numeric weather features
    num_cols = ["wind_speed_10m", "wind_direction_10m", "wind_gusts_10m", "temperature_2m", "surface_pressure"]
    raw_df[num_cols] = raw_df[num_cols].interpolate(method="linear").bfill().ffill()

    # 2. Dynamic air density calculation: rho = P / (R * T)
    raw_df["air_density_kg_m3"] = calculate_air_density(
        raw_df["surface_pressure"], raw_df["temperature_2m"]
    )

    # 3. Wind Power Density (WPD) in W/m^2: WPD = 0.5 * rho * v^3
    # Note: Computing hourly and then averaging preserves the cubic velocity variance
    raw_df["wpd_w_m2"] = 0.5 * raw_df["air_density_kg_m3"] * (raw_df["wind_speed_10m"] ** 3)

    # 4. Turbine Operability Status (True if within cut-in and cut-out threshold)
    raw_df["is_operable"] = (
        (raw_df["wind_speed_10m"] >= CUT_IN_SPEED_MS) &
        (raw_df["wind_speed_10m"] <= CUT_OUT_SPEED_MS)
    )

    # 5. Aggregate metrics per coordinate
    avg_speed = float(raw_df["wind_speed_10m"].mean())
    max_speed = float(raw_df["wind_speed_10m"].max())
    p90_speed = float(raw_df["wind_speed_10m"].quantile(0.90))
    avg_gust = float(raw_df["wind_gusts_10m"].mean())
    avg_wpd = float(raw_df["wpd_w_m2"].mean())
    operable_pct = float((raw_df["is_operable"].sum() / len(raw_df)) * 100.0)
    avg_temp = float(raw_df["temperature_2m"].mean())
    avg_density = float(raw_df["air_density_kg_m3"].mean())
    prevailing_dir = vector_mean_wind_direction(raw_df["wind_direction_10m"])
    cardinal_dir = get_compass_bearing(prevailing_dir)

    summary = {
        "site_id": site_id,
        "site_name": name,
        "category": category,
        "latitude": round(lat, 5),
        "longitude": round(lon, 5),
        "avg_wind_speed_10m": round(avg_speed, 2),
        "max_wind_speed_10m": round(max_speed, 2),
        "p90_wind_speed_10m": round(p90_speed, 2),
        "avg_gust_speed_10m": round(avg_gust, 2),
        "avg_wpd_w_m2": round(avg_wpd, 2),
        "operable_hours_pct": round(operable_pct, 2),
        "avg_temp_c": round(avg_temp, 2),
        "avg_air_density_kg_m3": round(avg_density, 3),
        "prevailing_wind_dir_deg": prevailing_dir,
        "prevailing_wind_dir_cardinal": cardinal_dir,
        "total_recorded_hours": len(raw_df),
        "time_window": f"{start_date} to {end_date}"
    }

    return summary, raw_df



# 6. MAIN EXECUTION PIPELINE

def main():
    print("=" * 80)
    print("AERO-GRID: AI Member 1 Atmospheric Data Extraction Pipeline")
    print("Target Region: VIT Bhopal / Kothrikalan / Ashta / Sehore Rural Corridor")
    print(f"Target Period: {START_DATE} to {END_DATE} (2 Full Calendar Years)")
    print(f"Total Candidate Siting Locations: {len(CANDIDATE_SITES)}")
    print("=" * 80)

    summaries: List[Dict] = []
    raw_datasets: List[pd.DataFrame] = []

    for site in CANDIDATE_SITES:
        try:
            summary, raw_df = process_site(site, START_DATE, END_DATE)
            summaries.append(summary)
            raw_datasets.append(raw_df)
            # Polite pause to stay well within Open-Meteo fair-use limits
            time.sleep(1.0)
        except Exception as err:
            print(f"[!] Failed to fetch site {site['site_id']}: {err}")

    if not summaries:
        print("[X] Extraction failed: No site data could be fetched.")
        return

    # Build master DataFrames
    summary_df = pd.DataFrame(summaries)
    raw_master_df = pd.concat(raw_datasets, ignore_index=True)

    # 1. Export Clean Averaged CSV (Primary deliverable for AI Member 2 & 3)
    summary_path = os.path.join(DATA_DIR, "site_weather_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\n[+] SUCCESS: Clean averaged CSV exported to:")
    print(f"    -> {summary_path}")

    # 2. Export Raw Hourly Dataset (For AI Member 3 time-series / Weibull distribution)
    raw_path = os.path.join(DATA_DIR, "site_weather_hourly_raw.csv")
    raw_master_df.to_csv(raw_path, index=False)
    print(f"[+] SUCCESS: Full hourly dataset exported to:")
    print(f"    -> {raw_path}")

    # Display preview table for verification
    print("\n" + "=" * 80)
    print("CLEAN AVERAGED CSV PREVIEW (Ready for AI Member 2 Spatial Join):")
    print("=" * 80)
    display_cols = [
        "site_id", "site_name", "latitude", "longitude",
        "avg_wind_speed_10m", "avg_wpd_w_m2", "operable_hours_pct",
        "prevailing_wind_dir_cardinal"
    ]
    print(summary_df[display_cols].to_string(index=False))
    print("=" * 80)
    print("[*] Next Step: AI Member 2 (Spatial/GIS) should merge elevation and land-use")
    print("    onto 'site_weather_summary.csv' matching on latitude & longitude.")
    print("=" * 80)


if __name__ == "__main__":
    main()
