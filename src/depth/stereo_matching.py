"""
Stereo Matching Module
Implements classical stereo matching algorithms for dense disparity estimation.
Includes multiple matching costs (SAD, SSD, NCC) and post-processing methods.
"""

import numpy as np
import cv2
from typing import Tuple, Optional
from tqdm import tqdm
from scipy.ndimage import median_filter


class StereoMatcher:
    """
    Classical stereo matching using block matching with various cost functions.
    
    Implemented cost functions:
    - SAD (Sum of Absolute Differences)
    - SSD (Sum of Squared Differences)
    - NCC (Normalized Cross-Correlation)
    
    Post-processing methods:
    - Left-right consistency check
    - Median filtering
    - Hole filling (interpolation)
    """
    
    def __init__(self, 
                 window_size: int = 11,
                 max_disparity: int = 128,
                 cost_function: str = 'SAD',
                 use_opencv: bool = True):
        """
        Initialize stereo matcher.
        
        Args:
            window_size: Size of matching window (odd number)
            max_disparity: Maximum disparity to search
            cost_function: 'SAD', 'SSD', 'NCC', or 'SGBM'
            use_opencv: Use OpenCV's optimized methods (much faster)
        """
        if window_size % 2 == 0:
            raise ValueError("Window size must be odd")
        
        self.window_size = window_size
        self.max_disparity = max_disparity
        self.cost_function = cost_function.upper()
        self.use_opencv = use_opencv
        
        if self.cost_function not in ['SAD', 'SSD', 'NCC', 'SGBM']:
            raise ValueError(f"Unknown cost function: {cost_function}")
    
    def compute_disparity(self, 
                         left_img: np.ndarray, 
                         right_img: np.ndarray,
                         verbose: bool = True) -> np.ndarray:
        """
        Compute disparity map using block matching.
        
        Args:
            left_img: Left rectified image (grayscale)
            right_img: Right rectified image (grayscale)
            verbose: Show progress bar
        
        Returns:
            Disparity map (same size as input images)
        """
        # Use fast OpenCV implementation by default
        if self.use_opencv:
            return self._compute_disparity_opencv(left_img, right_img, verbose)
        else:
            return self._compute_disparity_slow(left_img, right_img, verbose)
    
    def _compute_disparity_opencv(self,
                                  left_img: np.ndarray,
                                  right_img: np.ndarray, 
                                  verbose: bool = True) -> np.ndarray:
        """
        Fast disparity computation using OpenCV's optimized C++ implementations.
        Approximately 100-1000x faster than Python implementation.
        
        Args:
            left_img: Left rectified image
            right_img: Right rectified image
            verbose: Show progress (not used for OpenCV methods)
        
        Returns:
            Disparity map
        """
        # Convert to grayscale if needed
        if len(left_img.shape) == 3:
            left_img = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
        if len(right_img.shape) == 3:
            right_img = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
        
        # Convert to uint8 for OpenCV stereo matchers
        left_img = left_img.astype(np.uint8)
        right_img = right_img.astype(np.uint8)
        
        # Warning: OpenCV's StereoBM only supports SAD!
        if self.cost_function not in ['SAD', 'SGBM']:
            print(f"⚠️ WARNING: OpenCV StereoBM only supports SAD, not {self.cost_function}")
            print(f"⚠️ For correct {self.cost_function} results, use: use_opencv=False")
            print(f"⚠️ Falling back to SAD (OpenCV StereoBM)...\n")
        
        if verbose:
            actual_method = self.cost_function if self.cost_function in ['SAD', 'SGBM'] else f"{self.cost_function} (using SAD)"
            print(f"Computing disparity using OpenCV ({actual_method})...")
        
        if self.cost_function == 'SGBM':
            # Semi-Global Block Matching - best quality
            stereo = cv2.StereoSGBM_create(
                minDisparity=0,
                numDisparities=self.max_disparity,
                blockSize=self.window_size,
                P1=8 * 3 * self.window_size ** 2,
                P2=32 * 3 * self.window_size ** 2,
                disp12MaxDiff=1,
                uniquenessRatio=10,
                speckleWindowSize=100,
                speckleRange=32,
                mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
            )
        else:
            # Block Matching (SAD/SSD) - faster but lower quality
            stereo = cv2.StereoBM_create(
                numDisparities=self.max_disparity,
                blockSize=self.window_size
            )
            stereo.setPreFilterCap(31)
            stereo.setMinDisparity(0)
            stereo.setTextureThreshold(10)
            stereo.setUniquenessRatio(15)
            stereo.setSpeckleWindowSize(100)
            stereo.setSpeckleRange(32)
        
        # Compute disparity
        disparity = stereo.compute(left_img, right_img)
        
        # Convert from fixed-point (divide by 16)
        disparity = disparity.astype(np.float32) / 16.0
        
        # Set invalid disparities to 0
        disparity[disparity < 0] = 0
        
        if verbose:
            print(f"✓ Disparity computed using OpenCV (very fast!)")
        
        return disparity
    
    def _compute_disparity_slow(self, 
                         left_img: np.ndarray, 
                         right_img: np.ndarray,
                         verbose: bool = True) -> np.ndarray:
        """
        Original slow Python implementation of block matching.
        Kept for educational purposes and algorithm understanding.
        Use use_opencv=False to enable this.
        
        Args:
            left_img: Left rectified image (grayscale)
            right_img: Right rectified image (grayscale)
            verbose: Show progress bar
        
        Returns:
            Disparity map (same size as input images)
        """
        # Convert to grayscale if needed
        if len(left_img.shape) == 3:
            left_img = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
        if len(right_img.shape) == 3:
            right_img = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
        
        # Convert to float for computation
        left_img = left_img.astype(np.float32)
        right_img = right_img.astype(np.float32)
        
        h, w = left_img.shape
        half_win = self.window_size // 2
        
        # Initialize disparity map
        disparity = np.zeros((h, w), dtype=np.float32)
        
        # Choose cost function
        if self.cost_function == 'SAD':
            cost_fn = self._compute_sad
        elif self.cost_function == 'SSD':
            cost_fn = self._compute_ssd
        else:  # NCC
            cost_fn = self._compute_ncc
        
        # Compute disparity for each pixel
        iterator = tqdm(range(half_win, h - half_win), desc=f"Computing disparity ({self.cost_function})") if verbose else range(half_win, h - half_win)
        
        for y in iterator:
            for x in range(half_win, w - half_win):
                # Extract left window
                left_window = left_img[y - half_win:y + half_win + 1,
                                       x - half_win:x + half_win + 1]
                
                # Search along epipolar line (horizontal)
                best_disparity = 0
                best_cost = float('inf') if self.cost_function != 'NCC' else -float('inf')
                
                # Limit search range to valid image region
                min_d = 0
                max_d = min(self.max_disparity, x - half_win)
                
                for d in range(min_d, max_d):
                    # Extract right window shifted by disparity
                    right_window = right_img[y - half_win:y + half_win + 1,
                                             x - d - half_win:x - d + half_win + 1]
                    
                    # Compute cost
                    cost = cost_fn(left_window, right_window)
                    
                    # Update best disparity
                    if self.cost_function == 'NCC':
                        if cost > best_cost:
                            best_cost = cost
                            best_disparity = d
                    else:
                        if cost < best_cost:
                            best_cost = cost
                            best_disparity = d
                
                disparity[y, x] = best_disparity
        
        return disparity
    
    def _compute_sad(self, window1: np.ndarray, window2: np.ndarray) -> float:
        """
        Compute Sum of Absolute Differences.
        
        Args:
            window1: First image window
            window2: Second image window
        
        Returns:
            SAD cost
        """
        return np.sum(np.abs(window1 - window2))
    
    def _compute_ssd(self, window1: np.ndarray, window2: np.ndarray) -> float:
        """
        Compute Sum of Squared Differences.
        
        Args:
            window1: First image window
            window2: Second image window
        
        Returns:
            SSD cost
        """
        return np.sum((window1 - window2) ** 2)
    
    def _compute_ncc(self, window1: np.ndarray, window2: np.ndarray) -> float:
        """
        Compute Normalized Cross-Correlation.
        
        Args:
            window1: First image window
            window2: Second image window
        
        Returns:
            NCC cost (higher is better)
        """
        # Normalize windows
        w1 = window1 - np.mean(window1)
        w2 = window2 - np.mean(window2)
        
        # Compute correlation
        numerator = np.sum(w1 * w2)
        denominator = np.sqrt(np.sum(w1 ** 2) * np.sum(w2 ** 2))
        
        if denominator == 0:
            return -1.0
        
        return numerator / denominator
    
    def left_right_consistency_check(self, 
                                    left_disparity: np.ndarray,
                                    right_disparity: np.ndarray,
                                    threshold: float = 1.0) -> np.ndarray:
        """
        Perform left-right consistency check to detect occlusions.
        
        A pixel is marked as valid if:
        |d_left(x) - d_right(x - d_left(x))| < threshold
        
        Args:
            left_disparity: Disparity map from left to right
            right_disparity: Disparity map from right to left
            threshold: Consistency threshold in pixels
        
        Returns:
            Filtered disparity map (invalid pixels set to -1)
        """
        h, w = left_disparity.shape
        consistent_disparity = np.copy(left_disparity)
        
        for y in range(h):
            for x in range(w):
                d = int(left_disparity[y, x])
                
                # Check if corresponding point is within image bounds
                if x - d < 0 or x - d >= w:
                    consistent_disparity[y, x] = -1
                    continue
                
                # Check consistency
                d_right = right_disparity[y, x - d]
                if np.abs(d - d_right) > threshold:
                    consistent_disparity[y, x] = -1
        
        return consistent_disparity
    
    def median_filter_disparity(self, 
                                disparity: np.ndarray,
                                kernel_size: int = 5) -> np.ndarray:
        """
        Apply median filter to reduce speckle noise.
        
        Args:
            disparity: Input disparity map
            kernel_size: Size of median filter kernel
        
        Returns:
            Filtered disparity map
        """
        # Create mask for valid disparities
        valid_mask = disparity > 0
        
        # Apply median filter only to valid regions
        filtered = median_filter(disparity, size=kernel_size)
        
        # Restore invalid regions
        filtered[~valid_mask] = disparity[~valid_mask]
        
        return filtered
    
    def fill_holes(self, disparity: np.ndarray) -> np.ndarray:
        """
        Fill holes (invalid pixels) using simple interpolation.
        
        Invalid pixels are filled by nearest valid pixel in the same row.
        
        Args:
            disparity: Disparity map with holes (marked as -1 or 0)
        
        Returns:
            Disparity map with filled holes
        """
        h, w = disparity.shape
        filled = np.copy(disparity)
        
        for y in range(h):
            # Find valid and invalid pixels in this row
            valid_mask = filled[y, :] > 0
            invalid_mask = ~valid_mask
            
            if not np.any(valid_mask):
                continue
            
            # Get indices
            valid_indices = np.where(valid_mask)[0]
            invalid_indices = np.where(invalid_mask)[0]
            
            # Fill each invalid pixel with nearest valid value
            for x in invalid_indices:
                # Find nearest valid pixel
                distances = np.abs(valid_indices - x)
                nearest_idx = valid_indices[np.argmin(distances)]
                filled[y, x] = filled[y, nearest_idx]
        
        return filled
    
    def post_process(self,
                    left_img: np.ndarray,
                    right_img: np.ndarray,
                    left_disparity: np.ndarray,
                    apply_consistency: bool = True,
                    apply_median: bool = True,
                    apply_fill: bool = True,
                    consistency_threshold: float = 1.0,
                    median_kernel: int = 5,
                    verbose: bool = True) -> np.ndarray:
        """
        Apply post-processing to disparity map.
        
        Args:
            left_img: Left image
            right_img: Right image
            left_disparity: Disparity map to post-process
            apply_consistency: Apply left-right consistency check
            apply_median: Apply median filtering
            apply_fill: Apply hole filling
            consistency_threshold: Threshold for consistency check
            median_kernel: Kernel size for median filter
            verbose: Print progress
        
        Returns:
            Post-processed disparity map
        """
        disparity = np.copy(left_disparity)
        
        if apply_consistency:
            if verbose:
                print("Computing right disparity for consistency check...")
            # Compute right-to-left disparity
            right_disparity = self.compute_disparity(right_img, left_img, verbose=False)
            
            if verbose:
                print("Applying left-right consistency check...")
            disparity = self.left_right_consistency_check(
                disparity, right_disparity, consistency_threshold
            )
        
        if apply_median:
            if verbose:
                print("Applying median filter...")
            disparity = self.median_filter_disparity(disparity, median_kernel)
        
        if apply_fill:
            if verbose:
                print("Filling holes...")
            disparity = self.fill_holes(disparity)
        
        return disparity


def compute_disparity_opencv(left_img: np.ndarray,
                             right_img: np.ndarray,
                             method: str = 'SGBM',
                             **kwargs) -> np.ndarray:
    """
    Compute disparity using OpenCV's built-in stereo matchers.
    This is provided for comparison but should not be used as primary method.
    
    Args:
        left_img: Left rectified image
        right_img: Right rectified image
        method: 'BM' or 'SGBM'
        **kwargs: Additional parameters for stereo matcher
    
    Returns:
        Disparity map
    """
    # Convert to grayscale if needed
    if len(left_img.shape) == 3:
        left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
    else:
        left_gray = left_img
        right_gray = right_img
    
    if method == 'BM':
        # Block Matching
        num_disparities = kwargs.get('numDisparities', 128)
        block_size = kwargs.get('blockSize', 15)
        
        stereo = cv2.StereoBM_create(numDisparities=num_disparities,
                                     blockSize=block_size)
    else:
        # Semi-Global Block Matching
        num_disparities = kwargs.get('numDisparities', 128)
        block_size = kwargs.get('blockSize', 11)
        
        stereo = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=num_disparities,
            blockSize=block_size,
            P1=8 * 3 * block_size ** 2,
            P2=32 * 3 * block_size ** 2,
            disp12MaxDiff=1,
            uniquenessRatio=10,
            speckleWindowSize=100,
            speckleRange=32
        )
    
    # Compute disparity
    disparity = stereo.compute(left_gray, right_gray).astype(np.float32) / 16.0
    
    # Remove invalid disparities
    disparity[disparity < 0] = 0
    
    return disparity
