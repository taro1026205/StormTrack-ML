import numpy as np
import cv2
from PIL import Image
import xgboost as xgb
import pickle
import os
from scipy.ndimage import maximum_filter, minimum_filter

class MLPredictor:
    """
    XGBoost-based predictor for converting 2D radar PNG images into physical dBZ matrices.
    
    This class leverages a fast Look-Up Table (LUT) mapping coupled with a trained XGBoost
    residual model to infer true radar reflectivity (dBZ) values from flat RGB pixel data.
    """
    
    def __init__(self, model_path="models/xgboost_residual.json", lut_path="models/lut_config.pkl"):
        """
        Initializes the MLPredictor by precomputing spatial grids and loading model assets.
        
        Args:
            model_path (str): File path to the trained XGBoost JSON model.
            lut_path (str): File path to the pickled LUT configuration.
        """
        self.is_loaded = False
        self.model = None
        self.lut = None
        self.colors = None
        self.values = None
        self.fast_lut = None
        
        # Precompute spatial coordinate matrices (660x660 grid)
        self.X_GRID, self.Y_GRID = np.meshgrid(np.arange(660), np.arange(660))
        
        # Precompute polar coordinates relative to the radar center (330, 330)
        self.DIST_MAP = np.sqrt((self.X_GRID - 330)**2 + (self.Y_GRID - 330)**2)
        self.ANGLE_MAP = np.arctan2(self.Y_GRID - 330, self.X_GRID - 330)
        
        self.load_model(model_path, lut_path)
        
    def load_model(self, model_path, lut_path):
        """
        Loads the precomputed LUT mappings and the XGBoost regressor model.
        """
        if os.path.exists(lut_path):
            with open(lut_path, "rb") as f:
                lut_config = pickle.load(f)
                self.lut = lut_config["lut"]
                self.colors = lut_config["colors"]
                self.values = lut_config["values"]
                self.fast_lut = lut_config["fast_lut"]
                
        if os.path.exists(model_path):
            self.model = xgb.XGBRegressor()
            self.model.load_model(model_path)
            self.is_loaded = True
            
    def _apply_lut(self, img_rgb):
        """
        Applies a direct Color-to-dBZ mapping using a highly optimized 1D integer array.
        Falls back to a cKDTree Nearest-Neighbor search for unknown RGB variations.
        
        Args:
            img_rgb (np.ndarray): The RGB image matrix of shape (H, W, 3).
            
        Returns:
            np.ndarray: The baseline inferred dBZ map before XGBoost residual correction.
        """
        r, g, b = img_rgb[:,:,0].astype(int), img_rgb[:,:,1].astype(int), img_rgb[:,:,2].astype(int)
        
        # Calculate unique 1D index for RGB tuple (R*65536 + G*256 + B)
        idx = r * 65536 + g * 256 + b
        result = self.fast_lut[idx]
        nan_mask = np.isnan(result)
        
        # Resolve unmapped colors using spatial clustering (cKDTree)
        if np.any(nan_mask):
            flat_nan = img_rgb[nan_mask]
            if not hasattr(self, 'tree'):
                from scipy.spatial import cKDTree
                self.tree = cKDTree(self.colors)
                
            _, best_match = self.tree.query(flat_nan)
            matched_values = self.values[best_match]
            result[nan_mask] = matched_values
            
            # Cache newly resolved colors into the fast LUT for future O(1) lookups
            r_nan, g_nan, b_nan = flat_nan[:,0], flat_nan[:,1], flat_nan[:,2]
            idx_nan = (r_nan * 65536 + g_nan * 256 + b_nan).astype(int)
            self.fast_lut[idx_nan] = matched_values
            
        return result

    def predict_single_png_features(self, png_bytes, prev_clean_lut=None):
        """
        Extracts complex spatial and temporal features from a single radar image frame.
        
        Args:
            png_bytes: File-like object representing the PNG image.
            prev_clean_lut (np.ndarray, optional): The baseline dBZ map from the previous frame (for temporal delta).
            
        Returns:
            tuple: (X_test feature matrix, validity mask, baseline LUT guess, cleaned LUT matrix)
        """
        img = np.array(Image.open(png_bytes)).astype(np.float32)
        rgb, a = img[:, :, :3], img[:, :, 3]
        
        # 1. Obtain baseline dBZ estimate from colors
        lut_guess = self._apply_lut(rgb)
        
        # 2. Construct Validity Mask (Ignore background, text labels, and map overlays)
        r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
        valid_color_mask = (a > 0) & ~((r > 245) & (g > 245) & (b > 245) & (a == 255)) & ~((r < 10) & (g < 10) & (b < 10) & (a == 255))
        
        # 3. Clean background for robust feature engineering
        clean_lut = lut_guess.copy()
        clean_lut[~valid_color_mask] = 0.0
        
        # Compute temporal derivative (change in reflectivity over time)
        temp_delta = np.zeros_like(clean_lut) if prev_clean_lut is None else clean_lut - prev_clean_lut
        
        # 4. Generate Computer Vision Filters (Spatial Convolutions)
        blur_5 = cv2.GaussianBlur(clean_lut, (5, 5), 0)
        blur_31 = cv2.GaussianBlur(clean_lut, (31, 31), 0)
        max_5 = maximum_filter(clean_lut, size=5)
        min_5 = minimum_filter(clean_lut, size=5)
        log_lut = np.log1p(np.maximum(clean_lut, 0))
        laplacian = cv2.Laplacian(clean_lut, cv2.CV_32F)
        std_31 = np.sqrt(np.maximum(cv2.GaussianBlur(clean_lut**2, (31, 31), 0) - cv2.GaussianBlur(clean_lut, (31, 31), 0)**2, 0))
        edge = np.sqrt(cv2.Sobel(clean_lut, cv2.CV_32F, 1, 0, ksize=3)**2 + cv2.Sobel(clean_lut, cv2.CV_32F, 0, 1, ksize=3)**2)
        
        # 5. Assemble final multidimensional feature vector for XGBoost
        X_test = np.column_stack([
            self.X_GRID[valid_color_mask], self.Y_GRID[valid_color_mask], self.DIST_MAP[valid_color_mask], self.ANGLE_MAP[valid_color_mask],
            lut_guess[valid_color_mask], log_lut[valid_color_mask], blur_5[valid_color_mask], blur_31[valid_color_mask],
            (blur_5 - blur_31)[valid_color_mask], min_5[valid_color_mask], (max_5 - min_5)[valid_color_mask],
            max_5[valid_color_mask], edge[valid_color_mask], laplacian[valid_color_mask], std_31[valid_color_mask], temp_delta[valid_color_mask]
        ])
        
        return X_test, valid_color_mask, lut_guess, clean_lut
        
    def predict_pngs(self, uploaded_files):
        """
        Executes end-to-end Machine Learning inference on a sequence of PNG files.
        
        Args:
            uploaded_files (list): Sequence of file-like objects (e.g., IO streams).
            
        Returns:
            np.ndarray: A 3D matrix (Time, Y, X) containing the final reconstructed dBZ values.
        """
        all_X, masks, luts = [], [], []
        prev_clean_lut = None
        
        # 1. Feature Extraction Phase (Sequential for temporal delta)
        for file in uploaded_files:
            if hasattr(file, 'seek'): 
                file.seek(0)
            X_test, valid_color_mask, lut_guess, clean_lut = self.predict_single_png_features(file, prev_clean_lut)
            if X_test.shape[0] > 0: 
                all_X.append(X_test)
            masks.append(valid_color_mask)
            luts.append(lut_guess)
            prev_clean_lut = clean_lut
            
        if not all_X: 
            return np.full((len(uploaded_files), 660, 660), np.nan, dtype=np.float32)
            
        all_X_concat = np.vstack(all_X)
        
        # 2. Bulk Inference Phase (Highly optimized bulk prediction)
        all_residuals = self.model.predict(all_X_concat)
        
        # 3. Matrix Reconstruction Phase
        matrices = []
        pointer = 0
        
        for i in range(len(uploaded_files)):
            valid_color_mask = masks[i]
            lut_guess = luts[i]
            num_valid = np.sum(valid_color_mask)
            dbz_map = np.full((660, 660), np.nan, dtype=np.float32)
            
            if num_valid > 0:
                residual_pred = all_residuals[pointer : pointer + num_valid]
                pointer += num_valid
                
                # Final prediction = Baseline Color Mapping + XGBoost Residual Correction
                final_pred = lut_guess[valid_color_mask] + residual_pred
                
                # Heuristic amplification for extreme convection cores
                lut_valid = lut_guess[valid_color_mask]
                extreme_mask = (lut_valid >= 45)
                final_pred[extreme_mask] += 1.5
                
                dbz_map[valid_color_mask] = final_pred
                
            matrices.append(dbz_map)
            
        return np.stack(matrices, axis=0)
