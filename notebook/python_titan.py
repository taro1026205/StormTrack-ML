import numpy as np
import pandas as pd
from scipy.ndimage import label, distance_transform_edt
from skimage.measure import regionprops
from skimage.segmentation import watershed
from scipy.optimize import linear_sum_assignment

def identify_storms_dual_threshold(frame, dx_km, dy_km, low_thresh, high_thresh, min_area_km2):
    """
    Simulates the core TITAN (Thunderstorm Identification, Tracking, Analysis, and Nowcasting)
    Dual-Threshold Storm Identification algorithm using Watershed segmentation.
    
    The algorithm operates in three distinct phases:
    1. Extracts the broader storm envelope by thresholding at low_thresh.
    2. Identifies intense convective cores by thresholding at high_thresh.
    3. Leverages the Watershed algorithm to logically partition the larger envelope 
       into distinct storm cells governed by their respective convective cores.
       
    Args:
        frame (np.ndarray): 2D reflectivity matrix (dBZ).
        dx_km (float): Grid spacing in the X-direction (km).
        dy_km (float): Grid spacing in the Y-direction (km).
        low_thresh (float): Minimum dBZ to be considered part of a storm (e.g., 35 dBZ).
        high_thresh (float): Minimum dBZ to form a valid convective core (e.g., 45 dBZ).
        min_area_km2 (float): The minimum spatial area required to classify as a storm.
        
    Returns:
        tuple: (List of extracted storm features [dicts], 2D integer array of segmented labels)
    """
    envelope_mask = frame >= low_thresh
    core_mask = frame >= high_thresh
    
    # Isolate and label intense convective cores
    core_labels, num_cores = label(core_mask)
    
    if num_cores > 0:
        # Standard LROSE TITAN methodology: assign each point within the envelope 
        # to the nearest valid convective core using distance transformation.
        dist = distance_transform_edt(~core_mask)
        storm_labels = watershed(dist, core_labels, mask=envelope_mask)
    else:
        # Strict dual-thresholding dictates that an envelope lacking any intense core 
        # is discarded as noise or non-convective strata.
        storm_labels = np.zeros_like(frame, dtype=int)
        
    scan_features = []
    
    for region in regionprops(storm_labels, intensity_image=frame):
        area_km2 = region.area * dx_km * dy_km
        
        # Spatial thresholding: eliminate transient noise cells
        if area_km2 >= min_area_km2:
            max_dbz = np.max(region.intensity_image[region.image])
            
            # Intensity threshold validation for the entire contiguous region
            if high_thresh is not None and max_dbz < high_thresh:
                continue
                
            cy, cx = region.centroid
            
            # Map pixel coordinates to physical Cartesian space (meters)
            # Assumption: The radar origin (0,0) is located at the center of the grid matrix.
            y_m = (cy - frame.shape[0] / 2) * dy_km * 1000
            x_m = (cx - frame.shape[1] / 2) * dx_km * 1000
            
            # Compute equivalent spherical radius based on area (used for downstream rendering)
            radius_m = np.sqrt(area_km2 / np.pi) * 1000
            
            scan_features.append({
                'label': region.label,
                'centroid': (x_m, y_m),
                'radius': radius_m,
                'area': area_km2,
                'max_dbz': max_dbz
            })
            
    return scan_features, storm_labels


def track_storms_titan(cappi_data, times, dx_km=1.0, dy_km=1.0,
                       dbz_threshold=35.0, dual_threshold=45.0, min_area_km2=50.0):
    """
    Executes the comprehensive TITAN tracking algorithm across a chronological sequence of radar frames.
    
    This implementation utilizes Combinatorial Optimization (Hungarian algorithm) coupled with 
    First-Order Velocity Prediction (spatial advection projection) to robustly match storm cells 
    between consecutive scans, even amidst variable data acquisition rates.
    
    Args:
        cappi_data (np.ndarray): 3D Volume of radar data (Time, Y, X).
        times (list/array): Chronological timestamps corresponding to the temporal dimension.
        dx_km, dy_km (float): Grid spatial resolution.
        dbz_threshold (float): Primary threshold for identifying the storm envelope.
        dual_threshold (float): Secondary threshold for core identification.
        min_area_km2 (float): Minimum area constraint.
        
    Returns:
        tuple: (pd.DataFrame containing full storm trajectory metadata, 
                3D np.ndarray holding the sequential integer-labelled segmentation masks)
    """
    records = []
    titan_mask = np.zeros_like(cappi_data, dtype=int)
    
    # Phase 1: Exhaustive Storm Identification across all temporal frames
    scans_storms = []
    scans_labels = []
    
    for t_idx in range(cappi_data.shape[0]):
        frame = cappi_data[t_idx]
        features, storm_labels = identify_storms_dual_threshold(
            frame, dx_km, dy_km, dbz_threshold, dual_threshold, min_area_km2
        )
        scans_storms.append(features)
        scans_labels.append(storm_labels)
        
    # Phase 2: Trajectory Linking via Combinatorial Assignment
    current_cell_id = 1
    active_tracks = [] # Schema: {'id', 'last_pos', 'last_time_idx', 'velocity', 'last_area'}
    MAX_SPEED = 25.0 # Absolute velocity threshold (meters/second)
    
    for t_idx, storms in enumerate(scans_storms):
        # Dynamically calculate delta-T to handle irregular radar scan intervals
        dt_seconds = (times[t_idx] - times[t_idx-1]).total_seconds() if t_idx > 0 else 600
        max_dist = MAX_SPEED * dt_seconds
        
        n_tracks = len(active_tracks)
        n_storms = len(storms)
        
        # Initialize Cost Matrix with effectively infinite penalties
        cost_matrix = np.full((n_tracks, n_storms), 1e9)
        
        for i, track in enumerate(active_tracks):
            # Linearly advect the storm's last known position using its derived velocity vector
            vx, vy = track.get('velocity', (0.0, 0.0))
            pred_x = track['last_pos'][0] + vx * dt_seconds
            pred_y = track['last_pos'][1] + vy * dt_seconds
            
            for j, s in enumerate(storms):
                # Calculate Euclidean error distance from the projected forecast position
                dist = np.sqrt((s['centroid'][0] - pred_x)**2 + (s['centroid'][1] - pred_y)**2)
                
                # Evaluate morphological continuity (area expansion/contraction)
                area_ratio = min(track['last_area'], s['area']) / max(track['last_area'], s['area'])
                
                # Apply validation heuristics: constrain matching radius and max morphological variance (10x limit)
                if dist <= max_dist and area_ratio > 0.1:
                    # Multi-objective TITAN Cost Function: 
                    # Heavily weighted toward spatial proximity, with secondary size-variance penalties.
                    cost = dist + 1000 * (1.0 - area_ratio) 
                    cost_matrix[i, j] = cost
                    
        # Solve the Bipartite Matching Problem minimizing global assignment cost
        if n_tracks > 0 and n_storms > 0:
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
        else:
            row_ind, col_ind = [], []
        
        next_active_tracks = []
        assigned_storms = set()
        
        for i, j in zip(row_ind, col_ind):
            # Process strictly valid assignments (ignoring infinite cost blocks)
            if cost_matrix[i, j] < 1e8:
                s = storms[j]
                track = active_tracks[i]
                
                # Dynamically update the velocity vector for future advection steps
                vx = (s['centroid'][0] - track['last_pos'][0]) / dt_seconds
                vy = (s['centroid'][1] - track['last_pos'][1]) / dt_seconds
                
                # Synthesize a bounding polygon (circle approximation) for simplified downstream visualization
                angles = np.linspace(0, 2*np.pi, 72, endpoint=False)
                poly_x = s['centroid'][0] + s['radius'] * np.sin(angles)
                poly_y = s['centroid'][1] + s['radius'] * np.cos(angles)
                
                records.append({
                    'time': times[t_idx],
                    'cell': track['id'],
                    'x': s['centroid'][0],
                    'y': s['centroid'][1],
                    'poly_x': poly_x.tolist(),
                    'poly_y': poly_y.tolist(),
                    'area': s['area'],
                    'max_dbz': s.get('max_dbz', 0.0)
                })
                
                # Mutate track state for the next temporal iteration
                track['last_pos'] = s['centroid']
                track['velocity'] = (vx, vy)
                track['last_area'] = s['area']
                track['last_time_idx'] = t_idx
                next_active_tracks.append(track)
                assigned_storms.add(j)
                
                # Map the globally unique Cell ID back onto the 3D segmentation mask
                titan_mask[t_idx][scans_labels[t_idx] == s['label']] = track['id']
                
        # Phase 3: Instantiate new tracks for any newly emerged, unassigned storms
        for j, s in enumerate(storms):
            if j not in assigned_storms:
                new_id = current_cell_id
                
                angles = np.linspace(0, 2*np.pi, 72, endpoint=False)
                poly_x = s['centroid'][0] + s['radius'] * np.sin(angles)
                poly_y = s['centroid'][1] + s['radius'] * np.cos(angles)
                
                records.append({
                    'time': times[t_idx],
                    'cell': new_id,
                    'x': s['centroid'][0],
                    'y': s['centroid'][1],
                    'poly_x': poly_x.tolist(),
                    'poly_y': poly_y.tolist(),
                    'area': s['area'],
                    'max_dbz': s.get('max_dbz', 0.0)
                })
                
                next_active_tracks.append({
                    'id': new_id,
                    'last_pos': s['centroid'],
                    'velocity': (0.0, 0.0), # Unknown initial velocity upon genesis
                    'last_area': s['area'],
                    'last_time_idx': t_idx
                })
                titan_mask[t_idx][scans_labels[t_idx] == s['label']] = new_id
                current_cell_id += 1
                
        active_tracks = next_active_tracks
        
    df = pd.DataFrame(records)
    
    # Phase 4: Lifecycle Filtering (Standard TITAN heuristic requires >= 3 scans to be classified as a valid storm)
    if not df.empty:
        track_lengths = df.groupby('cell').size()
        valid_cells = track_lengths[track_lengths >= 3].index
        df = df[df['cell'].isin(valid_cells)]
        
    return df, titan_mask
