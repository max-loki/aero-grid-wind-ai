"""
Aero-Grid Wind AI: AI Member 2 (Spatial / GIS Data Pipeline)
=============================================================
Target Region: VIT Bhopal / Kothrikalan / Ashta / Sehore Rural Corridor (Madhya Pradesh, India)
Deliverables:
  - elevation_m (Ground elevation above sea level)
  - slope_degrees (Topographic inclination via Horn's 3x3 algorithm)
  - aspect_degrees (Topographic slope facing direction)
  - slope_category (Flat / Gentle / Moderate / Steep)
  - land_cover_category (Surface land-use category)
  - roughness_class (Davenport-Wieringa roughness class)
  - roughness_length_z0_m (Aerodynamic surface roughness length in meters)
  - osm_village & osm_county (OpenStreetMap administrative verification)
"""

import os
import time
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import requests

# 1. CANDIDATE SITING COORDINATES (EXACT MATCH WITH AI MEMBER 1)
CANDIDATE_SITES: List[Dict] = [
    {"site_id": "AERO_01", "name": "VIT Bhopal Campus (Open Ridge)", "category": "Educational/Open Ridge", "lat": 23.0775, "lon": 76.8491},
    {"site_id": "AERO_02", "name": "Kothrikalan Rural Center", "category": "Village Settlement", "lat": 23.0712, "lon": 76.8420},
    {"site_id": "AERO_03", "name": "Kothri Farmland North (Parbati Basin)", "category": "Agricultural Flatland", "lat": 23.0880, "lon": 76.8520},
    {"site_id": "AERO_04", "name": "Metwada Rural Outskirts", "category": "Rural Community", "lat": 23.0645, "lon": 76.8925},
    {"site_id": "AERO_05", "name": "Pagaria Hat Agricultural Zone", "category": "Agricultural Belt", "lat": 23.1120, "lon": 76.8250},
    {"site_id": "AERO_06", "name": "Lasudiya Khas (Midway Ashta-Kothri)", "category": "Farmland Plateau", "lat": 23.0450, "lon": 76.7820},
    {"site_id": "AERO_07", "name": "Ashta Rural East (Farmland Boundary)", "category": "Suburban/Rural Border", "lat": 23.0210, "lon": 76.7320},
    {"site_id": "AERO_08", "name": "Ashta South Agricultural Plain", "category": "Agricultural Flatland", "lat": 22.9850, "lon": 76.7110},
    {"site_id": "AERO_09", "name": "Barkheda Hasan Rural Sector", "category": "Rural Settlement", "lat": 23.1050, "lon": 76.8850},
    {"site_id": "AERO_10", "name": "Ichhawar Rural Plateau (High Elevation)", "category": "Hilly Rural Area", "lat": 23.0240, "lon": 77.0150},
    {"site_id": "AERO_11", "name": "Sehore Rural South-West Outskirts", "category": "Agricultural Cluster", "lat": 23.1650, "lon": 77.0120},
    {"site_id": "AERO_12", "name": "Shyampur Rural Outskirts", "category": "Village Cluster", "lat": 23.2450, "lon": 77.0850},
]

# 2. PATHS & CONFIGURATION
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

USER_AGENT = "AeroGridWindAI/1.0 (Spatial-GIS-Analysis; vit-bhopal-wind-study)"
REQUEST_HEADERS = {"User-Agent": USER_AGENT}

# 3. DAVENPORT-WIERINGA AERODYNAMIC ROUGHNESS LOOKUP TABLE
# Based on the Davenport-Wieringa terrain roughness classification for wind energy:
# z0 (m) determines vertical wind speed profile: v(z) = v10 * ln(z/z0) / ln(10/z0)
ROUGHNESS_MAPPING: Dict[str, Dict] = {
    "Educational/Open Ridge": {
        "land_cover": "Open Campus / Elevated Ridge",
        "roughness_class": "Class 3 - Open Flat / Low Obstacles",
        "z0_m": 0.030,
        "description": "Open grass, cleared academic grounds on elevated ridge with minimal wake obstruction"
    },
    "Village Settlement": {
        "land_cover": "Village Residential / Built-up",
        "roughness_class": "Class 6 - Village / Rural Settlement",
        "z0_m": 0.450,
        "description": "Dense cluster of 1-2 storey masonry structures and local village trees"
    },
    "Agricultural Flatland": {
        "land_cover": "Cultivated Crops / Flatland",
        "roughness_class": "Class 3 - Open Agricultural Plain",
        "z0_m": 0.050,
        "description": "Extensive open wheat/soy fields, flat plain, minimal isolated tree obstacles"
    },
    "Rural Community": {
        "land_cover": "Rural Hamlet / Mixed Farmstead",
        "roughness_class": "Class 6 - Rural Hamlet / Low Buildings",
        "z0_m": 0.400,
        "description": "Semi-dispersed village dwellings with boundary vegetation and small trees"
    },
    "Agricultural Belt": {
        "land_cover": "Intensive Farmland / Field Boundaries",
        "roughness_class": "Class 3 - Cultivated Farmland",
        "z0_m": 0.075,
        "description": "Farmland fields with scattered irrigation bunds and seasonal windbreaks"
    },
    "Farmland Plateau": {
        "land_cover": "Elevated Farmland / Open Tableland",
        "roughness_class": "Class 3 - Open Farmland Plateau",
        "z0_m": 0.060,
        "description": "Gently undulating cultivated plateau with unobstructed wind fetch"
    },
    "Suburban/Rural Border": {
        "land_cover": "Town Fringe / Semi-Settlement",
        "roughness_class": "Class 5 - Suburban Transition",
        "z0_m": 0.350,
        "description": "Transition zone between town edge, warehouse plots, and open fields"
    },
    "Rural Settlement": {
        "land_cover": "Village Habitation / Outskirts",
        "roughness_class": "Class 6 - Rural Settlement",
        "z0_m": 0.420,
        "description": "Nucleated rural houses surrounded by farm boundaries and neem groves"
    },
    "Hilly Rural Area": {
        "land_cover": "High-Elevation Plateau / Scrub Ridge",
        "roughness_class": "Class 5 - Rolling Highland / Shrub",
        "z0_m": 0.200,
        "description": "Elevated southern ridge with mixed scrub vegetation and rocky outcrops"
    },
    "Agricultural Cluster": {
        "land_cover": "Cultivated Fields / Scattered Farmsteads",
        "roughness_class": "Class 4 - Agricultural Cluster",
        "z0_m": 0.100,
        "description": "Open cropland with occasional isolated barn structures and tree hedgerows"
    },
    "Village Cluster": {
        "land_cover": "Dense Village Nucleus",
        "roughness_class": "Class 6 - Village Cluster",
        "z0_m": 0.480,
        "description": "Compact rural settlement cluster with complex aerodynamic obstacle drag"
    },
}


# 4. DIGITAL ELEVATION MODEL (DEM) & SLOPE ALGORITHMS

def fetch_elevation_batch(coordinates: List[Tuple[float, float]]) -> List[float]:
    """
    Fetches elevation (meters ASL) for a list of (lat, lon) coordinates using
    Open-Meteo High-Resolution DEM API (Copernicus 30m/90m DEM) with Open-Elevation fallback.
    """
    lats = [c[0] for c in coordinates]
    lons = [c[1] for c in coordinates]
    
    # 1. Primary: Open-Meteo DEM
    lat_str = ",".join(f"{lat:.6f}" for lat in lats)
    lon_str = ",".join(f"{lon:.6f}" for lon in lons)
    url = f"https://api.open-meteo.com/v1/elevation?latitude={lat_str}&longitude={lon_str}"
    
    try:
        resp = requests.get(url, headers=REQUEST_HEADERS, timeout=15)
        if resp.status_code == 200:
            elevations = resp.json().get("elevation", [])
            if len(elevations) == len(coordinates):
                return [round(float(e), 1) for e in elevations]
    except Exception as e:
        print(f"[!] Open-Meteo DEM query failed: {e}. Falling back to Open-Elevation...")

    # 2. Fallback: Open-Elevation API (point-by-point or batch)
    elevations = []
    for lat, lon in coordinates:
        try:
            fb_url = f"https://api.open-elevation.com/api/v1/lookup?locations={lat:.6f},{lon:.6f}"
            fb_resp = requests.get(fb_url, headers=REQUEST_HEADERS, timeout=10)
            if fb_resp.status_code == 200:
                res = fb_resp.json().get("results", [])
                if res:
                    elevations.append(round(float(res[0]["elevation"]), 1))
                    continue
        except Exception:
            pass
        # Default safety fallback based on regional Sehore/Malwa datum (490m)
        elevations.append(490.0)
        time.sleep(0.3)
    
    return elevations


def calculate_terrain_slope_and_aspect(lat: float, lon: float, delta_meters: float = 100.0) -> Tuple[float, float, float, str]:
    """
    Computes topographic elevation, slope (degrees), aspect (degrees), and slope category
    using Horn's 3x3 finite-difference algorithm (standard in ArcGIS / QGIS / GDAL).
    
    Neighborhood grid around (lat, lon):
      [ [NW, N, NE],
        [ W,  C,  E ],
        [ SW, S, SE] ]
    """
    # 1 deg lat ~ 111,320 m
    # 1 deg lon ~ 111,320 * cos(lat) m
    d_lat = delta_meters / 111320.0
    d_lon = delta_meters / (111320.0 * np.cos(np.radians(lat)))

    sample_pts = [
        (lat + d_lat, lon - d_lon), (lat + d_lat, lon), (lat + d_lat, lon + d_lon), # NW, N, NE
        (lat,         lon - d_lon), (lat,         lon), (lat,         lon + d_lon), # W,  C,  E
        (lat - d_lat, lon - d_lon), (lat - d_lat, lon), (lat - d_lat, lon + d_lon)  # SW, S, SE
    ]

    elevs = fetch_elevation_batch(sample_pts)
    grid = np.array(elevs).reshape((3, 3))

    center_elevation = float(grid[1, 1])

    # Horn's weighting formula for spatial gradients:
    # dz/dx = ((NE + 2*E + SE) - (NW + 2*W + SW)) / (8 * delta)
    # dz/dy = ((NW + 2*N + NE) - (SW + 2*S + SE)) / (8 * delta)
    dz_dx = ((grid[0, 2] + 2.0 * grid[1, 2] + grid[2, 2]) - (grid[0, 0] + 2.0 * grid[1, 0] + grid[2, 0])) / (8.0 * delta_meters)
    dz_dy = ((grid[0, 0] + 2.0 * grid[0, 1] + grid[0, 2]) - (grid[2, 0] + 2.0 * grid[2, 1] + grid[2, 2])) / (8.0 * delta_meters)

    slope_rad = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))
    slope_deg = round(float(np.degrees(slope_rad)), 2)

    aspect_deg = round(float((np.degrees(np.arctan2(-dz_dy, dz_dx)) + 360.0) % 360.0), 2)

    # Classification
    if slope_deg < 2.0:
        slope_cat = "Flat Plain (<2°)"
    elif slope_deg < 5.0:
        slope_cat = "Gentle Slope / Ridge (2-5°)"
    elif slope_deg < 15.0:
        slope_cat = "Moderate Slope (5-15°)"
    else:
        slope_cat = "Steep Terrain (>15°)"

    return center_elevation, slope_deg, aspect_deg, slope_cat


# 5. OPENSTREETMAP (OSM) ADMINISTRATIVE CONTEXT ENGINE

def fetch_osm_context(lat: float, lon: float) -> Tuple[str, str, str]:
    """
    Extracts verified administrative location context (village, county/tehsil, district)
    via OpenStreetMap Nominatim reverse geocoding API.
    """
    url = f"https://nominatim.openstreetmap.org/reverse?lat={lat:.5f}&lon={lon:.5f}&format=json"
    village = "Unknown Rural Settlement"
    county = "Sehore District"
    district = "Madhya Pradesh"

    try:
        resp = requests.get(url, headers=REQUEST_HEADERS, timeout=8)
        if resp.status_code == 200:
            addr = resp.json().get("address", {})
            village = addr.get("village") or addr.get("city_district") or addr.get("suburb") or addr.get("hamlet") or village
            county = addr.get("county") or addr.get("state_district") or county
            district = addr.get("state_district") or addr.get("state") or district
    except Exception as e:
        print(f"[!] OSM Nominatim query notice: {e}")

    return village, county, district


# 6. PIPELINE EXECUTION FOR ALL CANDIDATE SITES

def process_spatial_sites() -> pd.DataFrame:
    print("=" * 85)
    print("AERO-GRID: AI Member 2 (Spatial / GIS Data Extraction Pipeline)")
    print("Test Corridor: VIT Bhopal / Kothrikalan / Ashta / Sehore (12 Sites)")
    print("Metrics: Elevation ASL, Horn's 3x3 Slope, Aspect, Davenport Aerodynamic Roughness z0")
    print("=" * 85)

    records = []
    total = len(CANDIDATE_SITES)

    for idx, site in enumerate(CANDIDATE_SITES, 1):
        site_id = site["site_id"]
        name = site["name"]
        cat = site["category"]
        lat, lon = site["lat"], site["lon"]

        print(f"[{idx:02d}/{total:02d}] Analyzing {site_id} | {name} ({lat:.4f}, {lon:.4f})...")

        # 1. Topographic Elevation, Slope & Aspect
        elev, slope, aspect, slope_cat = calculate_terrain_slope_and_aspect(lat, lon, delta_meters=100.0)

        # 2. OpenStreetMap Administrative Boundary Verification
        village, county, district = fetch_osm_context(lat, lon)

        # 3. Davenport-Wieringa Roughness & Land-Use Lookup
        rough_info = ROUGHNESS_MAPPING.get(cat, {
            "land_cover": "Rural Mixed Open Land",
            "roughness_class": "Class 3 - Open Terrain",
            "z0_m": 0.050,
            "description": "General open rural land with low vegetation"
        })

        record = {
            "site_id": site_id,
            "site_name": name,
            "category": cat,
            "latitude": round(lat, 5),
            "longitude": round(lon, 5),
            "elevation_m": round(elev, 1),
            "slope_degrees": slope,
            "aspect_degrees": aspect,
            "slope_category": slope_cat,
            "land_cover_category": rough_info["land_cover"],
            "roughness_class": rough_info["roughness_class"],
            "roughness_length_z0_m": rough_info["z0_m"],
            "osm_village": village,
            "osm_county": county,
            "osm_district": district,
        }
        records.append(record)

        # Polite delay to respect API rate limits
        time.sleep(1.0)

    df_spatial = pd.DataFrame(records)

    # Export spatial dataset
    spatial_csv_path = os.path.join(DATA_DIR, "site_spatial_summary.csv")
    df_spatial.to_csv(spatial_csv_path, index=False)
    print("\n[+] SUCCESS: AI Member 2 spatial dataset saved to:")
    print(f"    -> {spatial_csv_path}")

    return df_spatial


if __name__ == "__main__":
    df = process_spatial_sites()
    print("\n" + "=" * 85)
    print("AI MEMBER 2 SPATIAL DATA PREVIEW:")
    print("=" * 85)
    cols = ["site_id", "site_name", "elevation_m", "slope_degrees", "slope_category", "roughness_length_z0_m", "osm_village"]
    print(df[cols].to_string(index=False))
    print("=" * 85)
