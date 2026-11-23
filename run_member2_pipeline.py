"""
Aero-Grid Wind AI: AI Member 2 Complete Pipeline Runner
========================================================
Runs:
  1. fetch_spatial_data.py -> Generates data/site_spatial_summary.csv
  2. merge_spatial_weather.py -> Generates data/site_master_dataset.csv
"""

import os
import sys

from fetch_spatial_data import process_spatial_sites
from merge_spatial_weather import merge_datasets

def main():
    print("=" * 85)
    print("STARTING AI MEMBER 2 (SPATIAL/GIS) END-TO-END PIPELINE")
    print("=" * 85)
    
    # Step 1: Spatial data extraction
    df_spatial = process_spatial_sites()
    
    # Step 2: Merge with Member 1's weather data
    df_master = merge_datasets()
    
    print("\n" + "=" * 85)
    print("AI MEMBER 2 PIPELINE COMPLETE!")
    print("Generated Deliverables:")
    print("  1. data/site_spatial_summary.csv  (Spatial / GIS standalone attributes)")
    print("  2. data/site_master_dataset.csv   (Integrated Master Dataset)")
    print("=" * 85)

if __name__ == "__main__":
    main()
