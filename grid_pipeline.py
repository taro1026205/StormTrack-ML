import pyart
import os
import glob
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

# ---------------------------------------------------------
# DIRECTORY CONFIGURATION
# ---------------------------------------------------------
RAW_DIR = '/Users/nguyenviethung/Desktop/Python-TITAN/file/data/raw'
GRID_DIR = '/Users/nguyenviethung/Desktop/Python-TITAN/file/data/grid'

os.makedirs(GRID_DIR, exist_ok=True)

# Traverse and chronologically index all RAW Polar NetCDF files
raw_files = sorted(glob.glob(os.path.join(RAW_DIR, '*_RAW.nc')))
print(f"Found {len(raw_files)} RAW files to process.")

# ---------------------------------------------------------
# SPATIAL GRID CONFIGURATION (3D CARTESIAN DOMAIN)
# ---------------------------------------------------------
# Z-Axis: 15 equidistant vertical levels (1km to 15km)
#         Essential for rendering accurate 3D bounding boxes and computing Echo Tops in TITAN.
# Y-Axis: 600km longitudinal sweep (-300km to +300km relative to radar origin), 1km resolution.
# X-Axis: 600km latitudinal sweep (-300km to +300km relative to radar origin), 1km resolution.
GRID_SHAPE = (15, 601, 601)

# Domain boundaries expressed in physical units (meters)
GRID_LIMITS = (
    (1000.0, 15000.0),      # Z limits: 1,000m to 15,000m
    (-300000.0, 300000.0),  # Y limits: -300km to +300km
    (-300000.0, 300000.0)   # X limits: -300km to +300km
)

print(f"Target Grid Shape (Z, Y, X): {GRID_SHAPE}")
print(f"Target Grid Limits: Z: 1-15km, Radius: 300km")

# ---------------------------------------------------------
# GRIDDING PIPELINE EXECUTION
# ---------------------------------------------------------
for file_path in tqdm(raw_files, desc="Gridding Radars"):
    filename = os.path.basename(file_path)
    out_filename = filename.replace('_RAW.nc', '_GRID.nc')
    out_path = os.path.join(GRID_DIR, out_filename)
    
    # Bypass processing if the heavily structured Volumetric NetCDF already exists
    if os.path.exists(out_path):
        continue
        
    try:
        # Step 1: Ingest the raw volumetric radar data (Polar sweeps)
        radar = pyart.io.read(file_path)
        
        # Step 2: Field selection constraint
        # Focus strictly on radar reflectivity; bypassing velocity/wind fields
        # to optimize memory and disk I/O.
        fields_to_keep = ['reflectivity']
        if 'total_power' in radar.fields and 'reflectivity' not in radar.fields:
            fields_to_keep = ['total_power']
            
        # Step 3: Spatial Interpolation (Polar to Cartesian transformation)
        # Employs the Barnes2 weighting function for robust smoothing and 
        # artifact mitigation during the mapping to the 600x600x15 grid.
        grid = pyart.map.grid_from_radars(
            (radar,),
            grid_shape=GRID_SHAPE,
            grid_limits=GRID_LIMITS,
            fields=fields_to_keep,
            gridding_algo='map_gates_to_grid',
            weighting_function='Barnes2'
        )
        
        # Step 4: Serialize the resulting 3D grid back to disk
        pyart.io.write_grid(out_path, grid)
        
    except Exception as e:
        print(f"Error processing {filename}: {e}")

print("Gridding Pipeline Completed Successfully!")
