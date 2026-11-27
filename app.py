"""
Aero-Grid | AI-Optimized Micro-Wind Turbine Siting Dashboard
=============================================================
Run with:  streamlit run app.py
"""

from __future__ import annotations

import math
import os
import sys

import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

# ── Ensure src/ is importable ────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from src.map_visuals import build_site_map
from evaluate_vawt_sites import (
    OPTIMIZED_VAWT,
    TERRAIN_ALPHA,
    adjust_wind_shear,
    map_category_to_terrain,
)

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="Aero-Grid | Micro-Wind Siting",
    page_icon="🌬️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Minimal CSS tweaks (keeps dark-theme config.toml in full control) ────────
st.markdown(
    """
    <style>
    /* KPI card styling */
    [data-testid="stMetric"] {
        background: #1A1F2B;
        border: 1px solid #2d3748;
        border-radius: 10px;
        padding: 16px 20px;
    }
    [data-testid="stMetricLabel"] { font-size: 0.78rem; color: #94a3b8 !important; }
    [data-testid="stMetricValue"] { font-size: 1.9rem; font-weight: 700; }

    /* Sidebar section headers */
    .sidebar-section {
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.08em;
        color: #00C896;
        text-transform: uppercase;
        margin: 18px 0 6px;
    }

    /* Warning / recommendation banners */
    .banner-warn {
        background: rgba(239,68,68,0.15);
        border-left: 4px solid #ef4444;
        border-radius: 6px;
        padding: 10px 14px;
        margin: 8px 0;
        font-size: 0.88rem;
        color: #fca5a5;
    }
    .banner-info {
        background: rgba(0,200,150,0.12);
        border-left: 4px solid #00C896;
        border-radius: 6px;
        padding: 10px 14px;
        margin: 8px 0;
        font-size: 0.88rem;
        color: #6ee7b7;
    }
    .banner-tip {
        background: rgba(249,115,22,0.13);
        border-left: 4px solid #f97316;
        border-radius: 6px;
        padding: 10px 14px;
        margin: 8px 0;
        font-size: 0.88rem;
        color: #fdba74;
    }

    /* Section dividers */
    hr { border-color: #2d3748 !important; margin: 1rem 0; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================================
# DATA LOADING & ENRICHMENT
# ============================================================================
CSV_PATH = os.path.join(os.path.dirname(__file__), "data", "site_master_dataset.csv")

# Turbine profiles ---------------------------------------------------------
TURBINE_PROFILES = {
    "500W (Standard HAWT)": {
        "rated_power_w": 500.0,
        "cut_in_ms": 4.0,
        "system_efficiency": 0.28,
        "capex_inr": 15000.0,
        "type": "HAWT",
    },
    "700W (Helical VAWT)": {
        "rated_power_w": 700.0,
        "cut_in_ms": 1.8,
        "system_efficiency": 0.20,
        "capex_inr": 15000.0,
        "type": "VAWT",
    },
    "1kW (Standard HAWT)": {
        "rated_power_w": 1000.0,
        "cut_in_ms": 4.0,
        "system_efficiency": 0.30,
        "capex_inr": 28000.0,
        "type": "HAWT",
    },
    "2kW (Standard HAWT)": {
        "rated_power_w": 2000.0,
        "cut_in_ms": 4.0,
        "system_efficiency": 0.32,
        "capex_inr": 55000.0,
        "type": "HAWT",
    },
}

TARIFF_INR_PER_KWH = 8.0          # ₹/kWh (from evaluate_vawt_sites.py)
KE = OPTIMIZED_VAWT["energy_pattern_factor"]   # Rayleigh factor = 1.91
H_REF = OPTIMIZED_VAWT["api_ref_height_m"]     # 10m reference height


@st.cache_data(show_spinner="Loading site dataset…")
def load_data(path: str) -> pd.DataFrame:
    """Load, clean, and enrich the master site dataset."""
    df = pd.read_csv(path)

    # ── Fix Â° encoding artifact ─────────────────────────────────────────────
    if "slope_category" in df.columns:
        df["slope_category"] = df["slope_category"].str.replace("Â°", "°", regex=False)

    # ── Derive suitability_score if absent ───────────────────────────────────
    if "suitability_score" not in df.columns:
        # Normalise each component to [0, 1] within the dataset
        wpd_norm   = (df["avg_wpd_w_m2"] - df["avg_wpd_w_m2"].min()) / (
                      df["avg_wpd_w_m2"].max() - df["avg_wpd_w_m2"].min() + 1e-9)
        opc_norm   = (df["operable_hours_pct"] - df["operable_hours_pct"].min()) / (
                      df["operable_hours_pct"].max() - df["operable_hours_pct"].min() + 1e-9)
        # Lower roughness → higher score (inverse)
        z0_inv     = 1.0 / (df["roughness_length_z0_m"] + 0.01)
        z0_norm    = (z0_inv - z0_inv.min()) / (z0_inv.max() - z0_inv.min() + 1e-9)
        # Weighted composite → scale to 0–100
        df["suitability_score"] = (
            0.55 * wpd_norm + 0.30 * opc_norm + 0.15 * z0_norm
        ) * 100.0

    # ── Derive viability_class if absent ─────────────────────────────────────
    if "viability_class" not in df.columns:
        def _classify(score: float) -> str:
            if score >= 70:
                return "High"
            elif score >= 45:
                return "Medium"
            return "Low"
        df["viability_class"] = df["suitability_score"].apply(_classify)

    return df


df_full = load_data(CSV_PATH)


# ============================================================================
# WIND-SHEAR HELPERS
# ============================================================================

def log_wind_profile(v_ref: float, z_ref: float, z_hub: float, z0: float) -> float:
    """Logarithmic wind profile extrapolation using roughness length."""
    if z0 <= 0:
        z0 = 0.001
    denom = math.log(z_ref / z0)
    if denom == 0:
        return v_ref
    return v_ref * (math.log(z_hub / z0) / denom)


def hellmann_wind_profile(v_ref: float, z_ref: float, z_hub: float, alpha: float) -> float:
    """Power-law (Hellmann) wind profile."""
    return adjust_wind_shear(v_ref, z_ref, z_hub, alpha)


def site_hub_wind(row: pd.Series, hub_height_m: float) -> float:
    """
    Scale 10m wind speed to hub height using the log profile first
    (uses per-site roughness_length_z0_m); falls back to Hellmann.
    """
    z0 = float(row.get("roughness_length_z0_m", 0.03))
    terrain = map_category_to_terrain(str(row.get("category", "")))
    alpha = TERRAIN_ALPHA.get(terrain, 0.28)
    v_10m = float(row["avg_wind_speed_10m"])
    # Use log profile when z0 is meaningful (>= 0.01 m); else Hellmann
    if z0 >= 0.01:
        return log_wind_profile(v_10m, H_REF, hub_height_m, z0)
    return hellmann_wind_profile(v_10m, H_REF, hub_height_m, alpha)


def compute_energy_metrics(
    v_hub: float,
    turbine: dict,
) -> dict:
    """
    Estimate daily/annual yield and payback using Rayleigh distribution factor.
    Mirrors evaluate_vawt_sites.evaluate_site() logic adapted for sidebar.
    """
    cut_in = turbine["cut_in_ms"]
    rated_w = turbine["rated_power_w"]
    eff = turbine["system_efficiency"]
    capex = turbine["capex_inr"]

    if v_hub < cut_in:
        avg_power_w = 0.0
    else:
        # WPD uses Rayleigh Ke factor (matches evaluate_vawt_sites)
        rho = 1.116   # representative air density for Sehore district ~495m
        wpd = 0.5 * rho * (v_hub ** 3) * KE
        swept_area = OPTIMIZED_VAWT["rotor_height_m"] * OPTIMIZED_VAWT["rotor_diameter_m"]
        raw_power_w = wpd * swept_area * eff
        avg_power_w = min(raw_power_w, rated_w)

    aep_kwh = (avg_power_w * 8760.0) / 1000.0
    daily_kwh = aep_kwh / 365.0
    annual_savings = aep_kwh * TARIFF_INR_PER_KWH
    payback_years = (capex / annual_savings) if annual_savings > 0 else float("inf")

    return {
        "avg_power_w": round(avg_power_w, 1),
        "daily_kwh": round(daily_kwh, 3),
        "aep_kwh": round(aep_kwh, 1),
        "annual_savings_inr": round(annual_savings, 0),
        "payback_years": round(payback_years, 1) if payback_years != float("inf") else None,
    }


# ============================================================================
# SIDEBAR
# ============================================================================
with st.sidebar:
    st.image(
        "https://img.icons8.com/fluency/96/wind-turbine.png",
        width=56,
    )
    st.title("Aero-Grid Controls")
    st.caption("AI-Optimized Micro-Wind Siting · Sehore District, MP")

    # ── SITE FILTERS ─────────────────────────────────────────────────────────
    st.markdown('<p class="sidebar-section">🔍 Site Filters</p>', unsafe_allow_html=True)

    all_classes = sorted(df_full["viability_class"].dropna().unique().tolist())
    sel_classes = st.multiselect(
        "Viability Class",
        options=all_classes,
        default=all_classes,
        help="Filter sites by their computed viability rating.",
    )

    min_ws_range = float(df_full["avg_wind_speed_10m"].min())
    max_ws_range = float(df_full["avg_wind_speed_10m"].max())
    min_ws = st.slider(
        "Min. Wind Speed @ 10m (m/s)",
        min_value=round(min_ws_range, 2),
        max_value=round(max_ws_range, 2),
        value=round(min_ws_range, 2),
        step=0.01,
        help="Show only sites with 10m mean wind speed ≥ this value.",
    )

    # ── HARDWARE & ENERGY ESTIMATOR ──────────────────────────────────────────
    st.markdown('<p class="sidebar-section">⚙️ Hardware & Energy Estimator</p>', unsafe_allow_html=True)

    turbine_key = st.selectbox(
        "Turbine Rating",
        options=list(TURBINE_PROFILES.keys()),
        index=1,   # default: 700W Helical VAWT
        help="Select turbine model; cut-in speed and economics adjust automatically.",
    )
    turbine = TURBINE_PROFILES[turbine_key]

    hub_height_label = st.radio(
        "Hub Height",
        options=["10m", "15m", "20m"],
        index=1,
        horizontal=True,
    )
    hub_height_m = float(hub_height_label.replace("m", ""))

    # ── Filtered DataFrame ────────────────────────────────────────────────────
    df_filtered = df_full[
        df_full["viability_class"].isin(sel_classes) &
        (df_full["avg_wind_speed_10m"] >= min_ws)
    ].copy()

    # Compute hub-height wind speed for every filtered site
    if not df_filtered.empty:
        df_filtered["hub_wind_speed_ms"] = df_filtered.apply(
            lambda r: round(site_hub_wind(r, hub_height_m), 2), axis=1
        )
    else:
        df_filtered["hub_wind_speed_ms"] = pd.Series(dtype=float)

    # ── Estimator: aggregate across filtered sites (mean site) ───────────────
    st.markdown("---")
    st.markdown("**Estimated Yield (Mean Site)**")

    if df_filtered.empty:
        st.warning("No sites match the current filters.")
        mean_v_hub = 0.0
        metrics = compute_energy_metrics(0.0, turbine)
    else:
        mean_v_hub = df_filtered["hub_wind_speed_ms"].mean()
        metrics = compute_energy_metrics(mean_v_hub, turbine)

    cut_in = turbine["cut_in_ms"]
    is_vawt = turbine["type"] == "VAWT"

    # Cut-in warning banner
    if mean_v_hub < cut_in and not df_filtered.empty:
        st.markdown(
            f'<div class="banner-warn">⚠️ Mean hub-height wind ({mean_v_hub:.2f} m/s) '
            f'is below the {turbine_key} cut-in speed ({cut_in} m/s). '
            f'Turbine will not generate power at average conditions.</div>',
            unsafe_allow_html=True,
        )
    elif not df_filtered.empty:
        col_a, col_b = st.columns(2)
        col_a.metric("Daily Yield", f"{metrics['daily_kwh']:.3f} kWh")
        col_b.metric("Annual Yield", f"{metrics['aep_kwh']:.0f} kWh")

        col_c, col_d = st.columns(2)
        col_c.metric("Annual Savings", f"₹{metrics['annual_savings_inr']:,.0f}")
        if metrics["payback_years"] is not None:
            col_d.metric("Payback Period", f"{metrics['payback_years']:.1f} yrs")
        else:
            col_d.metric("Payback Period", "∞")

    # ── VAWT vs. HAWT Recommendation ─────────────────────────────────────────
    if not df_filtered.empty:
        st.markdown("---")
        st.markdown("**Turbine Recommendation**")

        mean_z0 = df_filtered["roughness_length_z0_m"].mean()
        mean_gust_ratio = (
            df_filtered["avg_gust_speed_10m"] / df_filtered["avg_wind_speed_10m"]
        ).mean()

        recommend_vawt = (mean_z0 >= 0.3) or (mean_gust_ratio > 2.0)

        if recommend_vawt:
            st.markdown(
                f'<div class="banner-info">✅ <strong>Recommend: VAWT</strong><br>'
                f'Mean roughness z₀ = {mean_z0:.3f} m '
                f'{"(≥ 0.3 — turbulent terrain)" if mean_z0 >= 0.3 else ""}'
                f'{"· " if mean_z0 >= 0.3 and mean_gust_ratio > 2.0 else ""}'
                f'{"Gust ratio = " + f"{mean_gust_ratio:.2f}" + " (> 2.0)" if mean_gust_ratio > 2.0 else ""}'
                f'. VAWTs excel in turbulent, low-speed flows.</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="banner-tip">🔄 <strong>Either Turbine Suitable</strong><br>'
                f'Mean z₀ = {mean_z0:.3f} m · Gust ratio = {mean_gust_ratio:.2f}. '
                f'Open terrain favours HAWT; consider VAWT for rooftop or clustered-building sites.</div>',
                unsafe_allow_html=True,
            )

        if not is_vawt and recommend_vawt:
            st.markdown(
                '<div class="banner-warn">ℹ️ Your selected turbine is a HAWT, '
                'but site conditions suggest a VAWT would perform better.</div>',
                unsafe_allow_html=True,
            )

    st.markdown("---")
    st.caption(f"Dataset: {len(df_full)} total sites · Filtered: {len(df_filtered)}")


# ============================================================================
# MAIN CONTENT
# ============================================================================
# Header
st.markdown(
    """
    <div style="display:flex; align-items:center; gap:14px; margin-bottom:4px;">
        <span style="font-size:2.4rem;">🌬️</span>
        <div>
            <h1 style="margin:0; font-size:1.75rem; font-weight:800; color:#FAFAFA;">
                Aero-Grid
                <span style="color:#00C896; font-weight:900;">Wind Siting</span>
                Dashboard
            </h1>
            <p style="margin:0; color:#94a3b8; font-size:0.88rem;">
                AI-Optimized Micro-Wind Turbine Viability · Sehore District, Madhya Pradesh
            </p>
        </div>
    </div>
    <hr/>
    """,
    unsafe_allow_html=True,
)

# ── Empty-state guard ─────────────────────────────────────────────────────────
if df_filtered.empty:
    st.warning("⚠️ No sites match the current filters. Adjust the sidebar controls.")
    st.stop()

# ============================================================================
# KPI CARDS (Top Row)
# ============================================================================
total_sites   = len(df_filtered)
high_sites    = int((df_filtered["viability_class"] == "High").sum())
avg_hub_ws    = df_filtered["hub_wind_speed_ms"].mean()
max_wpd       = df_filtered["avg_wpd_w_m2"].max()

kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:
    st.metric(
        label="🗺️ Total Filtered Sites",
        value=total_sites,
        delta=f"{total_sites} of {len(df_full)} sites",
        delta_color="off",
    )
with kpi2:
    pct_high = int(high_sites / total_sites * 100) if total_sites else 0
    st.metric(
        label="✅ High-Viability Sites",
        value=high_sites,
        delta=f"{pct_high}% of filtered",
        delta_color="normal",
    )
with kpi3:
    st.metric(
        label=f"💨 Avg Hub Wind @ {hub_height_m}m",
        value=f"{avg_hub_ws:.2f} m/s",
        delta=f"ref: {df_filtered['avg_wind_speed_10m'].mean():.2f} m/s @ 10m",
        delta_color="off",
    )
with kpi4:
    st.metric(
        label="⚡ Max Wind Power Density",
        value=f"{max_wpd:.1f} W/m²",
        delta=f"mean: {df_filtered['avg_wpd_w_m2'].mean():.1f} W/m²",
        delta_color="off",
    )

st.markdown("<br/>", unsafe_allow_html=True)

# ============================================================================
# MAP SECTION
# ============================================================================
st.subheader("📍 Interactive Site Map")
st.caption(
    f"Showing **{total_sites}** sites · Hub height: **{hub_height_m}m** · "
    f"Turbine: **{turbine_key}** (cut-in: {turbine['cut_in_ms']} m/s)"
)

# Prepare map dataframe — rename wpd_wm2 → avg_wpd_w_m2 so map_visuals finds it
df_map = df_filtered.copy()
if "avg_wpd_w_m2" not in df_map.columns and "wpd_wm2" in df_map.columns:
    df_map = df_map.rename(columns={"wpd_wm2": "avg_wpd_w_m2"})
# Ensure viability_class is present for map_visuals (it resolves viability_class | viability)
if "viability_class" in df_map.columns and "viability" not in df_map.columns:
    df_map["viability"] = df_map["viability_class"]

try:
    folium_map = build_site_map(df_map, hub_height_m=hub_height_m)
    st_folium(folium_map, use_container_width=True, returned_objects=[], height=520)
except Exception as exc:
    st.error(f"Map rendering error: {exc}")

# ============================================================================
# DATA TABLE & EXPORT
# ============================================================================
st.markdown("<br/>", unsafe_allow_html=True)

with st.expander("📊 Explore Site Dataset", expanded=False):
    # Columns to display (ordered, human-friendly)
    display_cols = [
        col for col in [
            "site_id", "site_name", "osm_village", "osm_district",
            "viability_class", "suitability_score",
            "avg_wind_speed_10m", "hub_wind_speed_ms",
            "avg_wpd_w_m2", "operable_hours_pct",
            "prevailing_wind_dir_cardinal", "land_cover_category",
            "roughness_length_z0_m", "roughness_class",
            "elevation_m", "avg_temp_c",
            "avg_air_density_kg_m3", "avg_gust_speed_10m",
            "slope_category", "latitude", "longitude",
        ]
        if col in df_filtered.columns
    ]

    # Styling helpers
    def _color_viability(val: str) -> str:
        mapping = {
            "High": "color: #22c55e; font-weight: 700;",
            "Medium": "color: #f97316; font-weight: 700;",
            "Low": "color: #ef4444; font-weight: 700;",
        }
        return mapping.get(val, "")

    styled = (
        df_filtered[display_cols]
        .style
        .applymap(_color_viability, subset=["viability_class"])
        .format(
            {
                "suitability_score": "{:.1f}",
                "avg_wind_speed_10m": "{:.2f} m/s",
                "hub_wind_speed_ms": "{:.2f} m/s",
                "avg_wpd_w_m2": "{:.2f} W/m²",
                "operable_hours_pct": "{:.1f}%",
                "roughness_length_z0_m": "{:.3f} m",
                "elevation_m": "{:.0f} m",
                "avg_temp_c": "{:.1f} °C",
                "avg_air_density_kg_m3": "{:.3f}",
                "avg_gust_speed_10m": "{:.2f} m/s",
                "latitude": "{:.4f}",
                "longitude": "{:.4f}",
            },
            na_rep="—",
        )
    )

    st.dataframe(styled, use_container_width=True, height=320)

    # Export
    csv_bytes = df_filtered[display_cols].to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download Filtered Data as CSV",
        data=csv_bytes,
        file_name=f"aero_grid_filtered_{len(df_filtered)}sites.csv",
        mime="text/csv",
        help="Downloads the currently filtered site dataset.",
    )

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("<br/><br/>", unsafe_allow_html=True)
st.markdown(
    """
    <hr/>
    <p style="text-align:center; color:#475569; font-size:0.78rem;">
        Aero-Grid Wind AI · Sehore District VAWT Siting Study ·
        Data window: Jan 2023 – Dec 2024 · ERA5/Open-Meteo reanalysis ·
        Built with Streamlit + Folium
    </p>
    """,
    unsafe_allow_html=True,
)
