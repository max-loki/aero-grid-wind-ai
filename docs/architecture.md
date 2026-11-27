# Aero-Grid Wind AI — System Architecture

Aero-Grid is an end-to-end micro-wind turbine siting and techno-economic validation platform designed for low-wind, high-roughness rural and semi-urban environments.

## End-to-End Architecture Flowchart

```mermaid
flowchart LR
    %% Styling
    classDef ingestion fill:#1e293b,stroke:#38bdf8,stroke-width:2px,color:#f8fafc
    classDef pipeline fill:#1e293b,stroke:#a78bfa,stroke-width:2px,color:#f8fafc
    classDef validation fill:#1e293b,stroke:#f59e0b,stroke-width:2px,color:#f8fafc
    classDef ui fill:#064e3b,stroke:#00c896,stroke-width:2px,color:#ecfdf5
    classDef telemetry fill:#1f2937,stroke:#94a3b8,stroke-width:2px,stroke-dasharray: 5 5,color:#e2e8f0

    subgraph S1["1. Data Ingestion"]
        direction TB
        W["fetch_weather_data.py\n(Open-Meteo / NASA POWER API)"]:::ingestion
        SP["fetch_spatial_data.py\n(OpenStreetMap / Open-Elevation)"]:::ingestion
    end

    subgraph S2["2. Data Pipeline"]
        direction TB
        M["merge_spatial_weather.py"]:::pipeline
        CSV[("data/site_master_dataset.csv")]:::pipeline
        M --> CSV
    end

    subgraph S3["3. Physics, ML & CAD Validation"]
        direction TB
        EV["evaluate_vawt_sites.py\n(Helical VAWT Physics & Payback Model)"]:::validation
        CAD["Design/micro-wind.SLDPRT\n(Aerospace CFD & CAD Validation)"]:::validation
    end

    subgraph S4["4. Decision Support UI (Cloud)"]
        direction TB
        UI["app.py + src/map_visuals.py\n(Streamlit + Folium)"]:::ui
        CLOUD["Streamlit Community Cloud"]:::ui
        UI --> CLOUD
    end

    subgraph S5["5. Phase 2 Telemetry Loop (Future)"]
        direction TB
        EDGE["Field Anemometer + Microcontroller\n(Live Wind & Power Telemetry)"]:::telemetry
    end

    W --> M
    SP --> M
    CSV --> EV
    CSV --> UI
    EV --> UI
    CAD -.->|"Rotor Efficiency & Cut-In Specs"| EV
    CLOUD -.->|"Site Selection & Deployment"| EDGE
    EDGE -.->|"Live Wind & Power Stream"| CLOUD
```

## Stage Breakdown

| Stage | Component / Artifact | Responsibility |
| :--- | :--- | :--- |
| **1. Data Ingestion** | `fetch_weather_data.py`, `fetch_spatial_data.py` | Pulls multi-year wind speed, gust, temperature, and air-density time series from Open-Meteo / NASA POWER APIs alongside terrain elevation, slope, land cover, and surface roughness (`z₀`) from OpenStreetMap & Open-Elevation. |
| **2. Data Pipeline** | `merge_spatial_weather.py` → `data/site_master_dataset.csv` | Joins spatial terrain attributes with aggregated weather metrics into a unified, analysis-ready site master dataset. |
| **3. Physics, ML & CAD Validation** | `evaluate_vawt_sites.py`, `Design/micro-wind.SLDPRT` | Models Hellmann/log wind shear, Rayleigh wind power density, AEP, and INR payback for a 500W–700W Helical VAWT, informed by SolidWorks CAD & aerodynamic CFD validation. |
| **4. Decision Support UI (Cloud)** | `app.py`, `src/map_visuals.py` | Interactive Streamlit dashboard with Folium geospatial mapping, site suitability scoring, turbine hardware/hub-height estimator, and CSV export deployed on Streamlit Community Cloud. |
| **5. Phase 2 Telemetry Loop** | Field Anemometer + Microcontroller | *(Planned)* Edge sensor nodes deployed at top-ranked sites streaming real-time wind speed and power generation back to the cloud dashboard for model calibration and monitoring. |
