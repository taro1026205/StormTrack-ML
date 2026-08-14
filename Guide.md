# A Machine Learning & Algorithmic Storm Tracking Framework

This repository provides a comprehensive, end-to-end Python framework for identifying and tracking severe thunderstorms. It bridges the gap between traditional meteorological algorithms (Tobac, TITAN) and modern Deep Learning approaches (XGBoost Computer Vision), allowing for direct, apples-to-apples performance comparisons.

## Repository Structure

The project is structured into three main logical components: Data, Models, and Pipelines.

```text
Python_Storm_Tracking/
├── download_png.py                   # [Step 0] Radar image downloader script
├── grid_pipeline.py                  # [Step 1] Core data preprocessing script (RAW -> GRID)
├── file/
│   ├── big_event/                    # Historical severe storm event data & metadata 
│   ├── color/
│   │   └── GISS_isccp_rainbow.act    # Standard Panoply color palette for accurate dBZ mapping
│   ├── data/
│   │   ├── raw/                      # Original Polar NetCDF files from the radar station
│   │   ├── grid/                     # 3D Cartesian Volumetric Grids (Output of grid_pipeline.py)
│   │   └── png/                      # Rendered RGB radar images (used as ML input)
│   └── models/
│       ├── xgboost_residual.json     # Trained XGBoost model file
│       └── lut_config.pkl            # Look-Up Table (LUT) configuration for color mapping
├── notebook/
│   ├── cappi_hybrid_mapping.ipynb    # [Step 2] Model Training pipeline (NetCDF vs PNG)
│   ├── ml_predictor.py               # XGBoost inference wrapper (Converts PNG to 2D CAPPI Matrix)
│   ├── python_titan.py               # Python simulator of the LROSE TITAN Tracking Algorithm
│   ├── Tracking_Pipeline.ipynb       # [Step 3a] Ground Truth Tracking (Runs on RAW NetCDF Grid)
│   ├── ML_Tracking_Pipeline.ipynb    # [Step 3b] Machine Learning Tracking (Runs on ML Inferred Grid)
│   └── Big_Event_Visual_Pipeline.ipynb # [Step 4] Deep-Dive Analysis, Statistical Visualization & Tracking Comparison
└── references/                         # Contains foundational PDF scientific literature, project proposals.
```

---

## End-to-End Execution Guide

Follow this guide to successfully run the entire thunderstorm analysis and tracking pipeline from scratch.

### Step 0: Data Acquisition (Radar PNGs)
If you don't have the visual PNG images yet, you can download them directly from the weather web server.
1. Run the downloader script with your desired time range:
   ```bash
   python3 download_png.py --start "2022-08-15 00:00" --end "2022-08-15 23:50" --folder "file/data/png"
   ```
2. **Output:** The script will automatically fetch all available radar PNG images at 10-minute intervals and save them to the specified folder.

### Step 1: Data Preprocessing (Polar to Cartesian)
Radar stations export raw data in polar coordinates (rays, azimuths, elevations). To perform spatial tracking, we must convert this into a 3D Cartesian box (X, Y, Z).
1. Place your raw NetCDF files in `file/data/raw/`.
2. Run the Gridding Pipeline:
   ```bash
   python3 grid_pipeline.py
   ```
3. **Output:** The script uses `pyart` to generate heavily structured 3D Volumetric files (`*_GRID.nc`) inside the `file/data/grid/` folder. This is your definitive Ground Truth data.

### Step 2: Machine Learning Model Training
We want to teach an AI model how to look at a simple visual radar PNG image and reconstruct the precise physical dBZ matrix that usually only exists inside the heavy `GRID.nc` files.
1. Ensure your `png` images are in `file/data/png/` and your `GRID.nc` files are in `file/data/grid/`.
2. Open `notebook/cappi_hybrid_mapping.ipynb`.
3. Run the notebook from top to bottom.
4. **Output:** The notebook will extract 2D CAPPI layers from the NetCDF grids, map them against the PNG pixels, and train an XGBoost spatial model. The trained model will be saved to `file/models/`.

### Step 3: Storm Tracking & Comparison Pipelines
The core scientific value of this repository lies in comparing how well the algorithms track storms using pristine Ground Truth data versus AI-reconstructed data.

#### The Tracking Engines:
Both pipelines utilize two distinct tracking algorithms to ensure robustness:
- **Tobac:** An established framework using `scipy` thresholding and `trackpy`.
- **Python TITAN Simulator (`python_titan.py`):** A custom-built algorithm simulating the C++ LROSE TITAN engine. It uses **Dual-Threshold Watershed Segmentation** to split storm cores, and a **Combinatorial Hungarian Assignment** to optimize cost-based trajectory matching.

#### 3a. Ground Truth Pipeline (RAW)
1. Open `notebook/Tracking_Pipeline.ipynb`.
2. This notebook reads the exact physical data from `file/data/grid/` (extracting the 2D CAPPI layer).
3. It runs both Tobac and the TITAN Simulator.
4. **Output:** Reliable, physically accurate trajectories of storms.

#### 3b. Machine Learning Pipeline (AI)
1. Open `notebook/ML_Tracking_Pipeline.ipynb`.
2. This notebook completely ignores the NetCDF data! Instead, it imports `ml_predictor.py` to read the visual `PNG` files and uses the XGBoost model to instantly synthesize a virtual dBZ matrix.
3. The virtual matrix is resized and flipped to perfectly match the physical coordinate system of the raw data.
4. It then runs the *exact same* Tobac and TITAN Simulator engines used in the RAW pipeline.
5. **Output:** ML-inferred trajectories.

#### Step 4: Deep-Dive Statistical Analysis (Big Event)
Once the tracking is verified, we perform rigorous statistical evaluation on severe events (e.g., the massive 126mm rainfall event on 20220815).
1. Open `notebook/Big_Event_Visual_Pipeline.ipynb`.
2. This notebook computes comprehensive storm population statistics directly.
3. **Output:** 
   - Summary Tables of Storm Lifetimes, Areas, Intensities, and Velocities.
   - Kernel Density Estimate (KDE) Histograms.
   - Jittered Boxplots & Kinematic Wind Roses.
   - High-definition Cartopy animations spanning the entire 600x600 km radar coverage.
