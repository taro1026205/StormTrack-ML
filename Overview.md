# Python-TITAN: Machine Learning & Algorithmic Storm Tracking Pipeline

This repository contains the implementation of a **hybrid data science framework** for **storm tracking and atmospheric convection analysis using legacy radar data**.  
The proposed architecture integrates an **XGBoost Machine Learning pipeline** for spatial data reconstruction with **state-of-the-art Lagrangian tracking algorithms (TITAN and TOBAC)** to effectively capture storm morphologies, kinematics, and lifecycles over Ho Chi Minh City.

## Overview

Quantifying the impact of the Urban Heat Island (UHI) effect on local precipitation is a challenging task due to the lossy nature of historical radar archives (RGB PNGs) and algorithmic biases in storm tracking. To address this issue, our pipeline combines three complementary components:

1. **Machine Learning Radar Data Rescue (XGBoost)**
    - Extracts 16 spatial features (gradients, distances, edges) to reconstruct continuous, high-fidelity physical reflectivity (dBZ) matrices directly from compressed, anti-aliased 2D color images.

2. **3D Volumetric Gridding (Py-ART)**
    - Converts raw polar coordinate data (azimuth, elevation, range) into a standardized $15 \times 601 \times 601$ 3D Cartesian grid to serve as the absolute Ground Truth.

3. **Dual-Algorithm Tracking Evaluation**
    - **TITAN Simulator:** Utilizes Dual-Threshold Watershed Segmentation and Combinatorial Hungarian Optimization (preserving area and mass).
    - **TOBAC:** Utilizes Gaussian smoothing, multi-threshold topological feature detection, and kinematic nearest-neighbor linking (`trackpy`).

Additionally, **Advanced Statistical Visualizations** (KDE Histograms, Jittered Boxplots, and Kinematic Wind Roses) are used to provide **algorithmic interpretability**, highlighting how tracking formulations (Area vs. Kinematics) fundamentally dictate derived storm lifecycles, mergers, and splits.

---

## Architecture

The overall pipeline consists of the following steps:

1. Automated radar PNG acquisition and EDA (Exploratory Data Analysis).
2. XGBoost spatial feature learning and LUT (Look-Up Table) calibration.
3. 3D NetCDF volumetric data gridding via `pyart` and `xarray`.
4. Storm cell identification (Watershed vs. Topological).
5. Lagrangian trajectory linking (Hungarian cost-matrix vs. Kinematic Crocker-Grier).
6. Post-processing and synthesis of kinematic properties and peak intensities.

## Dataset

The model is designed to process and track **high-resolution weather radar datasets**.

Example dataset sources include:
- General `.nc` (NetCDF) radar volume scans and legacy `.png` visual archives.

> [!NOTE]
> **Data Privacy Notice:** The raw `.nc` files used to train the XGBoost reconstruction model are privately provided by the South Vietnam Regional Hydrometeorological Centre (SRHMC) and cannot be publicly distributed. This dataset exclusively contains the full radar volume scans for two specific dates: **November 6th, 2025** and **November 7th, 2025**.

## Contact

For questions or collaborations, please contact [hung.nguyensubin106@hcmut.edu.vn](mailto:hung.nguyensubin106@hcmut.edu.vn)
