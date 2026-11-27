"""
Aero-Grid Wind AI: Interactive Folium Map Visualizations
=========================================================
Provides `build_site_map(df, hub_height_m)` for the Streamlit dashboard.

Expected DataFrame columns (produced by evaluate_vawt_sites.run_pipeline()):
    latitude, longitude, site_name, osm_village,
    viability_class | viability  -- viability label (High / Medium / Low / Unsuitable)
    suitability_score           -- optional numeric score (shown when present)
    hub_wind_speed_ms           -- wind speed at hub height
    avg_wpd_w_m2 | wpd_wm2     -- wind power density W/m²
    prevailing_wind_dir_cardinal
    land_cover_category
"""

from __future__ import annotations

import folium
from folium import plugins
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Viability → (fill_color, border_color, icon_color, FontAwesome icon)
_VIABILITY_STYLE: dict[str, tuple[str, str, str, str]] = {
    "high":       ("#22c55e", "#15803d", "white", "bolt"),
    "medium":     ("#f97316", "#c2410c", "white", "wind"),
    "low":        ("#ef4444", "#b91c1c", "white", "exclamation-triangle"),
    "unsuitable": ("#6b7280", "#374151", "white", "ban"),
}

_DEFAULT_STYLE = ("#94a3b8", "#475569", "white", "question")

_TILE_LAYERS = [
    {
        "tiles": "CartoDB positron",
        "name": "CartoDB Positron (Light)",
        "attr": "© OpenStreetMap contributors © CartoDB",
    },
    {
        "tiles": "OpenStreetMap",
        "name": "OpenStreetMap",
        "attr": "© OpenStreetMap contributors",
    },
    {
        "tiles": "CartoDB dark_matter",
        "name": "CartoDB Dark Matter",
        "attr": "© OpenStreetMap contributors © CartoDB",
    },
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _resolve_col(row: pd.Series, *candidates: str, default="—") -> str:
    """Return the first non-null candidate column value as a string."""
    for col in candidates:
        if col in row.index and pd.notna(row[col]):
            val = row[col]
            return str(val)
    return default


def _viability_key(row: pd.Series) -> str:
    """Normalise viability label to lowercase key."""
    raw = _resolve_col(row, "viability_class", "viability", default="unknown")
    return raw.strip().lower()


def _make_popup_html(row: pd.Series, hub_height_m: float) -> str:
    """Build a styled HTML popup card for a site marker."""
    vkey = _viability_key(row)
    color_map = {
        "high": "#22c55e",
        "medium": "#f97316",
        "low": "#ef4444",
        "unsuitable": "#6b7280",
    }
    badge_color = color_map.get(vkey, "#94a3b8")
    viability_label = _resolve_col(row, "viability_class", "viability")

    site_name       = _resolve_col(row, "site_name")
    osm_village     = _resolve_col(row, "osm_village")
    suitability     = _resolve_col(row, "suitability_score")
    hub_ws          = _resolve_col(row, "hub_wind_speed_ms")
    wpd             = _resolve_col(row, "avg_wpd_w_m2", "wpd_wm2")
    wind_dir        = _resolve_col(row, "prevailing_wind_dir_cardinal")
    land_cover      = _resolve_col(row, "land_cover_category")

    # Format numeric fields nicely
    try:
        hub_ws = f"{float(hub_ws):.2f} m/s"
    except (ValueError, TypeError):
        hub_ws = hub_ws

    try:
        wpd = f"{float(wpd):.1f} W/m²"
    except (ValueError, TypeError):
        wpd = wpd

    suitability_row = ""
    if suitability != "—":
        try:
            suitability_row = (
                f'<tr><td style="color:#94a3b8">Suitability Score</td>'
                f'<td><strong>{float(suitability):.2f}</strong></td></tr>'
            )
        except (ValueError, TypeError):
            suitability_row = (
                f'<tr><td style="color:#94a3b8">Suitability Score</td>'
                f'<td><strong>{suitability}</strong></td></tr>'
            )

    return f"""
    <div style="
        font-family: 'Segoe UI', Arial, sans-serif;
        min-width: 240px;
        max-width: 300px;
        font-size: 13px;
        line-height: 1.6;
        color: #1e293b;
    ">
        <div style="
            background: {badge_color};
            color: white;
            padding: 8px 12px;
            border-radius: 6px 6px 0 0;
            font-weight: 700;
            font-size: 14px;
        ">
            {site_name}
        </div>
        <div style="
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-top: none;
            border-radius: 0 0 6px 6px;
            padding: 10px 12px;
        ">
            <table style="width:100%; border-collapse:collapse;">
                <tr>
                    <td style="color:#94a3b8; padding: 2px 0; width:50%">Village</td>
                    <td><strong>{osm_village}</strong></td>
                </tr>
                <tr>
                    <td style="color:#94a3b8; padding: 2px 0">Viability</td>
                    <td>
                        <span style="
                            background:{badge_color};
                            color:white;
                            padding:1px 7px;
                            border-radius:999px;
                            font-size:11px;
                            font-weight:600;
                        ">{viability_label}</span>
                    </td>
                </tr>
                {suitability_row}
                <tr>
                    <td style="color:#94a3b8; padding: 2px 0">Hub Wind Speed</td>
                    <td><strong>{hub_ws}</strong>
                        <span style="color:#94a3b8;font-size:11px"> @ {hub_height_m}m</span>
                    </td>
                </tr>
                <tr>
                    <td style="color:#94a3b8; padding: 2px 0">Wind Power Density</td>
                    <td><strong>{wpd}</strong></td>
                </tr>
                <tr>
                    <td style="color:#94a3b8; padding: 2px 0">Prevailing Wind</td>
                    <td><strong>{wind_dir}</strong></td>
                </tr>
                <tr>
                    <td style="color:#94a3b8; padding: 2px 0">Land Cover</td>
                    <td style="font-size:11px">{land_cover}</td>
                </tr>
            </table>
        </div>
    </div>
    """.strip()


def _make_tooltip(row: pd.Series) -> str:
    """Short one-line tooltip shown on hover."""
    site_name = _resolve_col(row, "site_name")
    viability = _resolve_col(row, "viability_class", "viability")
    hub_ws    = _resolve_col(row, "hub_wind_speed_ms")
    try:
        hub_ws = f"{float(hub_ws):.2f} m/s"
    except (ValueError, TypeError):
        pass
    return f"{site_name} · {viability} · {hub_ws}"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_site_map(df: pd.DataFrame, hub_height_m: float) -> folium.Map:
    """
    Build an interactive Folium map of wind assessment sites.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame produced by ``evaluate_vawt_sites.run_pipeline()``.
        Must contain at minimum: latitude, longitude, site_name.
        Viability colouring uses ``viability_class`` or ``viability``.
    hub_height_m : float
        Hub height (metres) used in labelling popup wind-speed values.

    Returns
    -------
    folium.Map
        Ready to render with::

            st_folium(m, use_container_width=True, returned_objects=[])
    """
    if df.empty:
        raise ValueError("build_site_map: received an empty DataFrame.")

    # ── Map centre ──────────────────────────────────────────────────────────
    center_lat = df["latitude"].mean()
    center_lon = df["longitude"].mean()

    # ── Base map (CartoDB Positron as default clean tile) ───────────────────
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=11,
        tiles=None,          # tiles added explicitly below for layer control
        control_scale=True,
    )

    # ── Tile layers ──────────────────────────────────────────────────────────
    for i, layer in enumerate(_TILE_LAYERS):
        folium.TileLayer(
            tiles=layer["tiles"],
            name=layer["name"],
            attr=layer["attr"],
            show=(i == 0),   # only the first tile layer is visible by default
        ).add_to(m)

    # ── Feature groups (one per viability class for layer control) ───────────
    groups: dict[str, folium.FeatureGroup] = {}
    group_order = ["High", "Medium", "Low", "Unsuitable", "Unknown"]
    for label in group_order:
        fg = folium.FeatureGroup(name=f"⬤ {label} Viability", show=True)
        groups[label] = fg
        fg.add_to(m)

    # ── Markers ──────────────────────────────────────────────────────────────
    for _, row in df.iterrows():
        vkey = _viability_key(row)
        fill_color, border_color, icon_color, fa_icon = _VIABILITY_STYLE.get(
            vkey, _DEFAULT_STYLE
        )
        viability_label = _resolve_col(row, "viability_class", "viability", default="Unknown")
        group_key = viability_label.capitalize() if viability_label.capitalize() in groups else "Unknown"
        if group_key not in groups:
            # Safety net: create an ad-hoc group
            fg = folium.FeatureGroup(name=f"⬤ {group_key} Viability", show=True)
            groups[group_key] = fg
            fg.add_to(m)

        popup_html = _make_popup_html(row, hub_height_m)
        tooltip_text = _make_tooltip(row)

        marker = folium.CircleMarker(
            location=[row["latitude"], row["longitude"]],
            radius=10,
            color=border_color,
            weight=2,
            fill=True,
            fill_color=fill_color,
            fill_opacity=0.85,
            popup=folium.Popup(
                folium.IFrame(popup_html, width=320, height=230),
                max_width=320,
            ),
            tooltip=folium.Tooltip(
                tooltip_text,
                style=(
                    "background-color: #1e293b;"
                    "color: white;"
                    "border: none;"
                    "border-radius: 4px;"
                    "font-family: 'Segoe UI', Arial, sans-serif;"
                    "font-size: 12px;"
                    "padding: 4px 8px;"
                ),
            ),
        )
        marker.add_to(groups[group_key])

    # ── Fullscreen plugin ────────────────────────────────────────────────────
    plugins.Fullscreen(
        position="topright",
        title="Enter fullscreen",
        title_cancel="Exit fullscreen",
        force_separate_button=True,
    ).add_to(m)

    # ── Mini-map ─────────────────────────────────────────────────────────────
    plugins.MiniMap(toggle_display=True, position="bottomright").add_to(m)

    # ── Layer control (must be added last) ───────────────────────────────────
    folium.LayerControl(position="topright", collapsed=False).add_to(m)

    return m
