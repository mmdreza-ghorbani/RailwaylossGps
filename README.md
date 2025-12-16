# Railway-Constrained GPS Trajectory Correction

## Overview

This project implements a **deterministic, railway-constrained GPS trajectory correction pipeline** designed originally for railway drive-test and telemetry data.

The goal is to **reconstruct accurate point locations when GPS data contains noise, spikes, missing seconds, teleports, or ordering issues**, while strictly respecting railway geometry and temporal continuity.

Instead of relying on heavy probabilistic map-matching frameworks, this approach focuses on **explainable, rule-based correction** that is suitable for post-processing, analytics, and audit-friendly environments (e.g., telecom drive tests, trajectory data).

---

## Key Characteristics

* Network-constrained (railway geometry aware)
* Anchor-based trajectory reconstruction (Good → Bad → Good)
* Time-aware interpolation
* Order and monotonicity enforcement along the rail
* Deterministic and explainable results
* Designed for batch/offline correction

---

## Problem Statement

Railway GPS data often suffers from:

* Missing timestamps (gaps in 1‑second sampling)
* GPS drift and spikes
* Teleporting points (unrealistic speeds)
* Points far from the actual railway
* Repeated or stuck coordinates
* Backward movement due to GPS jitter

Naive snapping of each point to the nearest rail segment leads to:

* Zig-zagging between parallel tracks
* Loss of movement continuity
* Incorrect ordering along the railway

This project addresses these issues holistically.

---

## High-Level Approach

### 1. Temporal Segmentation

* Input points are kept **in original row order** (to preserve Excel ↔ KML alignment).
* The trajectory is split into segments where consecutive timestamps differ by exactly 1 second.
* Each segment is processed independently.

---

### 2. Point Classification

Each point is classified using multiple rules:

* **Stuck points**: repeated coordinates
* **Spikes**: excessively far from railway
* **Far points**: beyond a maximum allowed rail distance
* **Teleports**: unrealistic speed between consecutive points
* **Time-order violations**: non-increasing timestamps

Points passing all checks are treated as **good anchors**.

---

### 3. Railway Geometry Preparation

* Railway shapefile is reprojected to EPSG:3857
* All rail geometries are merged and exploded into individual `LineString` components
* The longest rail component is selected as a **reference line** for order checking

---

### 4. Anchor-Based Reconstruction (G → B → G)

For each continuous segment:

* Consecutive **good anchor points** are identified
* A single railway `LineString` that best fits both anchors is selected
* Anchors are projected onto that same rail line
* All intermediate bad points are **interpolated along the rail** using timestamp fractions

This prevents:

* Switching between parallel tracks
* Independent snapping artifacts
* Geometric discontinuities

---

### 5. Order Enforcement

* All points are projected onto a reference railway line
* Backward movement along the rail (beyond a tolerance) is detected
* Violating points are iteratively removed from the anchor set

This ensures monotonic movement along the railway.

---

### 6. Edge Handling

* Leading bad points before the first anchor are held at the first anchor position
* Trailing bad points after the last anchor are held at the last anchor position
* Segments with no valid anchors are snapped point-by-point as a fallback

---

## Outputs

The pipeline produces:

* **Corrected KML files** (updated coordinates + timestamps)
* **Debug Excel file** with classification flags
* **Missing seconds report**
* **Optional virtual track geometry (GeoJSON)** when gaps exist
* **Interactive Folium map** for visual inspection

---

## Design Philosophy

* Deterministic > probabilistic
* Explainability over black-box models
* Domain-specific rules for railway movement
* Practical engineering over academic novelty

This approach intentionally avoids HMM or Kalman filtering in favor of transparency and controllability.

---

## Intended Use Cases

* Railway drive-test post-processing
* Telecom network KPI alignment with GPS
* Rail-based telemetry cleanup
* Offline trajectory reconstruction
* Engineering analysis and auditing

---

## Limitations

* Not intended for real-time processing
* Assumes reliable railway geometry
* Requires at least some valid anchor points per segment
* Optimized for rail, not free-road networks

---

## Terminology

* **Anchor**: A GPS point considered valid and trustworthy
* **Segment**: A run of points with continuous 1-second timestamps
* **s-coordinate**: Distance along a railway LineString

---

## License & Attribution

This project builds upon well-established GIS and trajectory reconstruction concepts, implemented here as a practical, domain-specific pipeline.

Feel free to reuse, adapt, or extend with appropriate attribution.

---

## Author

Developed as part of a real-world railway GPS correction and drive-test analysis workflow.
