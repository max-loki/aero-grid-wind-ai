"""
Aero-Grid Wind AI: VAWT Site Evaluation & Payback Pipeline
==========================================================
Evaluates 12 sites from data/site_master_dataset.csv for
a 500W-700W Helical Vertical-Axis Wind Turbine (VAWT).
"""

import os
import math
import pandas as pd

# =====================================================================
# OPTIMIZED VERTICAL-AXIS WIND TURBINE (500W-700W Helical VAWT)
# =====================================================================
OPTIMIZED_VAWT = {
    "rotor_height_m": 1.5,        # Helical Blade Height (H)
    "rotor_diameter_m": 1.0,      # Helical Blade Diameter (D)
    "hub_height_m": 15.0,         # Elevated mast / rooftop mount
    "api_ref_height_m": 10.0,     # Standard Weather API reference
    "cut_in_speed_ms": 1.8,       # Low-cogging coreless AFPMG start speed
    "rated_power_w": 700.0,       # Peak Generator Cap (W)
    "system_efficiency": 0.20,    # AFPMG generator + MPPT controller
    "energy_pattern_factor": 1.91,# Ke for Rayleigh distribution
    "capital_cost_inr": 15000.0,  # Value-engineered / local fabrication cost
    "tariff_inr_per_kwh": 8.0     # Realistic domestic tier 2 / commercial tariff
}

# Terrain roughness exponents (Hellmann alpha)
TERRAIN_ALPHA = {
    "open": 0.14,       # Flat farmland, open spaces
    "semi_open": 0.20,  # Scattered trees, rural towns
    "rough": 0.28,      # Dense trees, suburban campus (often ideal for VAWTs)
    "urban": 0.38       # Rooftops, built-up areas
}


def calculate_air_density(elevation_m: float, temp_c: float) -> float:
    """Calculates local air density using elevation pressure drop."""
    P0 = 101325.0       
    T0 = 288.15         
    L = 0.0065          
    g = 9.80665         
    M = 0.0289644       
    R_univ = 8.31447    
    R_spec = 287.058    

    pressure_pa = P0 * (1 - (L * elevation_m) / T0) ** ((g * M) / (R_univ * L))
    temp_k = temp_c + 273.15
    return pressure_pa / (R_spec * temp_k)


def adjust_wind_shear(v_ref: float, h_ref: float, h_hub: float, alpha: float) -> float:
    """Hellmann Law for vertical profile extrapolation."""
    return v_ref * ((h_hub / h_ref) ** alpha)


def evaluate_site(
    v_ref_ms: float,
    elevation_m: float,
    temp_c: float,
    terrain_type: str = "rough",
    specs: dict = OPTIMIZED_VAWT
) -> dict:
    """Evaluates site compatibility for Helical Vertical-Axis Turbine (VAWT)."""
    alpha = TERRAIN_ALPHA.get(terrain_type.lower(), 0.28)
    rho = calculate_air_density(elevation_m, temp_c)
    v_hub = adjust_wind_shear(
        v_ref_ms, specs["api_ref_height_m"], specs["hub_height_m"], alpha
    )

    swept_area = specs["rotor_height_m"] * specs["rotor_diameter_m"]
    wpd = 0.5 * rho * (v_hub ** 3) * specs["energy_pattern_factor"]

    if v_hub < specs["cut_in_speed_ms"]:
        avg_power_w = 0.0
    else:
        raw_elec_w = wpd * swept_area * specs["system_efficiency"]
        avg_power_w = min(raw_elec_w, specs["rated_power_w"])

    aep_kwh = (avg_power_w * 8760.0) / 1000.0
    annual_savings_inr = aep_kwh * specs["tariff_inr_per_kwh"]
    payback_years = (
        specs["capital_cost_inr"] / annual_savings_inr
        if annual_savings_inr > 0
        else float("inf")
    )

    if v_hub >= 4.5 and wpd >= 110:
        viability = "High"
    elif v_hub >= 3.2 and wpd >= 50:
        viability = "Medium"
    elif v_hub >= specs["cut_in_speed_ms"]:
        viability = "Low"
    else:
        viability = "Unsuitable"

    return {
        "air_density_kgm3": round(rho, 3),
        "hub_wind_speed_ms": round(v_hub, 2),
        "wpd_wm2": round(wpd, 2),
        "avg_power_w": round(avg_power_w, 1),
        "aep_kwh": round(aep_kwh, 1),
        "payback_years": round(payback_years, 2) if payback_years != float("inf") else None,
        "viability": viability
    }


def required_wind_speed_for_payback(
    target_months: float,
    elevation_m: float = 500.0,
    temp_c: float = 28.0,
    specs: dict = OPTIMIZED_VAWT
) -> float:
    """Calculates necessary mean wind speed (m/s) for ROI target."""
    rho = calculate_air_density(elevation_m, temp_c)
    swept_area = specs["rotor_height_m"] * specs["rotor_diameter_m"]

    e_req_kwh = specs["capital_cost_inr"] / specs["tariff_inr_per_kwh"]
    p_avg_req_w = (e_req_kwh * 1000.0) / (target_months * 730.0)

    denom = rho * swept_area * specs["system_efficiency"] * specs["energy_pattern_factor"]
    v_req = ((2.0 * p_avg_req_w) / denom) ** (1.0 / 3.0)
    return round(v_req, 2)


def map_category_to_terrain(category: str) -> str:
    """Maps dataset site category to Hellmann roughness key."""
    cat = str(category).lower()
    if any(k in cat for k in ["agricultural", "flatland"]):
        return "open"
    elif any(k in cat for k in ["ridge", "campus", "plateau"]):
        return "semi_open"
    elif any(k in cat for k in ["suburban", "village", "settlement", "community"]):
        return "rough"
    return "rough"


def run_pipeline(data_source: str = None) -> pd.DataFrame:
    # Use local file if present, otherwise fetch from GitHub
    local_csv = os.path.join(os.path.dirname(__file__), "data", "site_master_dataset.csv")
    if data_source:
        csv_path = data_source
    elif os.path.exists(local_csv):
        csv_path = local_csv
    else:
        csv_path = "https://raw.githubusercontent.com/max-loki/aero-grid-wind-ai/main/data/site_master_dataset.csv"

    print(f"[*] Loading dataset from: {csv_path}")
    df = pd.read_csv(csv_path)

    df["terrain"] = df["category"].apply(map_category_to_terrain)

    results = df.apply(
        lambda r: evaluate_site(
            v_ref_ms=r["avg_wind_speed_10m"],
            elevation_m=r["elevation_m"],
            temp_c=r["avg_temp_c"],
            terrain_type=r["terrain"]
        ),
        axis=1,
        result_type="expand"
    )

    final_df = pd.concat([df, results], axis=1)
    return final_df


if __name__ == "__main__":
    print("=" * 85)
    print("AERO-GRID: VAWT Site Viability & Payback Assessment")
    print("=" * 85)

    print("\n--- Required Wind Speeds for ROI Targets (OPTIMIZED_VAWT) ---")
    for yrs in [5, 10, 20, 30, 40]:
        months = yrs * 12
        print(f"To break even in {yrs:2d} years ({months:3d} mos): need {required_wind_speed_for_payback(months)} m/s")

    final_df = run_pipeline()

    cols_to_display = [
        "site_id", "site_name", "avg_wind_speed_10m",
        "hub_wind_speed_ms", "wpd_wm2", "avg_power_w",
        "aep_kwh", "payback_years", "viability"
    ]
    print("\n" + "=" * 85)
    print("SITE EVALUATION RESULTS:")
    print("=" * 85)
    print(final_df[cols_to_display].to_string(index=False))
    print("=" * 85)
