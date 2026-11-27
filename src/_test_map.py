"""Quick smoke-test for map_visuals.build_site_map."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from evaluate_vawt_sites import run_pipeline, OPTIMIZED_VAWT
from src.map_visuals import build_site_map
import folium

df = run_pipeline()
print("Cols:", list(df.columns))
print("Rows:", len(df))
print("viability values:", df["viability"].unique())

m = build_site_map(df, hub_height_m=OPTIMIZED_VAWT["hub_height_m"])
assert isinstance(m, folium.Map), f"Expected folium.Map, got {type(m)}"
print("Map location:", m.location)
print("SUCCESS: build_site_map returned a valid folium.Map")
