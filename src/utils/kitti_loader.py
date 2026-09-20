"""
KITTI Dataset Loader
Handles loading stereo images, calibration data, and ground truth information
from the KITTI dataset (Odometry and Stereo 2015 benchmarks).
"""

import numpy as np
import cv2
import os
from typing import Tuple, Dict, Optional


class KITTIOdometryLoader:
    """
    Loader for KITTI Odometry dataset.
    
    Dataset structure:
    data_odometry/
    ├── sequences/
    │   ├── 00/
    │   │   ├── image_0/  (left grayscale)
    │   │   ├── image_1/  (right grayscale)
    │   │   ├── image_2/  (left color)
    │   │   ├── image_3/  (right color)
    │   │   └── calib.txt
    │   └── ...
    └── poses/
        ├── 00.txt
        └── ...
    """
    
    def __init__(self, dataset_path: str, sequence: str = "00"):
        """
        Initialize KITTI Odometry loader.
        
        Args:
            dataset_path: Path to KITTI odometry dataset root
            sequence: Sequence number (00-21)
        """
        self.dataset_path = dataset_path
        self.sequence = sequence
        
        # Check if using standard KITTI structure or user's structure
        standard_path = os.path.join(dataset_path, "sequences", sequence)
        direct_path = os.path.join(dataset_path, sequence)
        
        if os.path.exists(standard_path):
            self.sequence_path = standard_path
        elif os.path.exists(direct_path):
            self.sequence_path = direct_path
        else:
            self.sequence_path = standard_path  # Default to standard
        
        # Image directories
        self.left_gray_dir = os.path.join(self.sequence_path, "image_0")
        self.right_gray_dir = os.path.join(self.sequence_path, "image_1")
        self.left_color_dir = os.path.join(self.sequence_path, "image_2")
        self.right_color_dir = os.path.join(self.sequence_path, "image_3")
        
        # Calibration file
        self.calib_file = os.path.join(self.sequence_path, "calib.txt")
        
        # Ground truth poses (only available for sequences 00-10)
        # Check both standard location and sequence folder
        poses_standard = os.path.join(dataset_path, "poses", f"{sequence}.txt")
        poses_in_seq = os.path.join(self.sequence_path, f"{sequence}.txt")
        
        if os.path.exists(poses_standard):
            self.poses_file = poses_standard
        elif os.path.exists(poses_in_seq):
            self.poses_file = poses_in_seq
        else:
            self.poses_file = poses_standard  # Default
        
        # Load calibration and poses
        self.calib = self._load_calibration()
        self.poses = self._load_poses() if os.path.exists(self.poses_file) else None
        
        # Count number of frames
        self.num_frames = len(os.listdir(self.left_gray_dir)) if os.path.exists(self.left_gray_dir) else 0
    
    def _load_calibration(self) -> Dict[str, np.ndarray]:
        """
        Load calibration data from calib.txt.
        
        Returns:
            Dictionary containing projection matrices P0, P1, P2, P3
        """
        calib = {}
        if not os.path.exists(self.calib_file):
            # Return default calibration if file doesn't exist
            print(f"Warning: Calibration file not found at {self.calib_file}")
            print("Using default calibration values")
            # Default values for KITTI sequence 00
            calib['P0'] = np.array([[718.856, 0, 607.1928, 0],
                                    [0, 718.856, 185.2157, 0],
                                    [0, 0, 1, 0]])
            calib['P1'] = np.array([[718.856, 0, 607.1928, -386.1448],
                                    [0, 718.856, 185.2157, 0],
                                    [0, 0, 1, 0]])
            calib['P2'] = np.array([[721.5377, 0, 609.5593, 44.85728],
                                    [0, 721.5377, 172.854, 0.2163791],
                                    [0, 0, 1, 0.002745884]])
            calib['P3'] = np.array([[721.5377, 0, 609.5593, -339.5242],
                                    [0, 721.5377, 172.854, 2.199936],
                                    [0, 0, 1, 0.002914259]])
            return calib
        
        with open(self.calib_file, 'r') as f:
            for line in f:
                key, value = line.split(':', 1)
                calib[key] = np.array([float(x) for x in value.split()])
        
        # Reshape projection matrices to 3x4
        for key in ['P0', 'P1', 'P2', 'P3']:
            if key in calib:
                calib[key] = calib[key].reshape(3, 4)
        
        return calib
    
    def _load_poses(self) -> Optional[np.ndarray]:
        """
        Load ground truth poses from poses file.
        
        Returns:
            Array of shape (N, 3, 4) containing camera poses
        """
        if not os.path.exists(self.poses_file):
            print(f"Warning: Poses file not found at {self.poses_file}")
            return None
        
        poses = []
        with open(self.poses_file, 'r') as f:
            for line in f:
                T = np.fromstring(line, dtype=float, sep=' ')
                T = T.reshape(3, 4)
                poses.append(T)
        
        return np.array(poses)
    
    def get_stereo_pair(self, frame_idx: int, color: bool = False) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load stereo image pair for given frame index.
        
        Args:
            frame_idx: Frame index
            color: If True, load color images; otherwise load grayscale
        
        Returns:
            Tuple of (left_image, right_image)
        """
        if color:
            left_dir = self.left_color_dir
            right_dir = self.right_color_dir
        else:
            left_dir = self.left_gray_dir
            right_dir = self.right_gray_dir
        
        left_path = os.path.join(left_dir, f"{frame_idx:06d}.png")
        right_path = os.path.join(right_dir, f"{frame_idx:06d}.png")
        
        left_img = cv2.imread(left_path, cv2.IMREAD_COLOR if color else cv2.IMREAD_GRAYSCALE)
        right_img = cv2.imread(right_path, cv2.IMREAD_COLOR if color else cv2.IMREAD_GRAYSCALE)
        
        if left_img is None or right_img is None:
            raise FileNotFoundError(f"Could not load images for frame {frame_idx}")
        
        return left_img, right_img
    
    def get_camera_params(self, camera: str = 'P0') -> Dict[str, float]:
        """
        Extract camera parameters from projection matrix.
        
        Args:
            camera: Which camera projection matrix to use ('P0', 'P1', 'P2', 'P3')
        
        Returns:
            Dictionary containing focal length, baseline, and principal point
        """
        P_left = self.calib[camera]
        
        # For stereo, use P0 (left grayscale) and P1 (right grayscale)
        # or P2 (left color) and P3 (right color)
        if camera in ['P0', 'P2']:
            if camera == 'P0':
                P_right = self.calib['P1']
            else:
                P_right = self.calib['P3']
            
            # Extract focal length (assuming fx = fy)
            fx = P_left[0, 0]
            fy = P_left[1, 1]
            
            # Principal point
            cx = P_left[0, 2]
            cy = P_left[1, 2]
            
            # Baseline (from translation component)
            # P1[0,3] = -fx * baseline
            baseline = np.abs(P_right[0, 3] / fx)
            
            return {
                'fx': fx,
                'fy': fy,
                'cx': cx,
                'cy': cy,
                'baseline': baseline
            }
        else:
            raise ValueError(f"Invalid camera: {camera}")
    
    def get_intrinsics(self, camera: str = 'P0') -> np.ndarray:
        """
        Get camera intrinsic matrix K.
        
        Args:
            camera: Which camera to extract intrinsics from
        
        Returns:
            3x3 intrinsic matrix
        """
        params = self.get_camera_params(camera)
        K = np.array([
            [params['fx'], 0, params['cx']],
            [0, params['fy'], params['cy']],
            [0, 0, 1]
        ])
        return K
    
    def get_pose(self, frame_idx: int) -> Optional[np.ndarray]:
        """
        Get ground truth pose for given frame.
        
        Args:
            frame_idx: Frame index
        
        Returns:
            4x4 transformation matrix (homogeneous coordinates)
        """
        if self.poses is None:
            return None
        
        if frame_idx >= len(self.poses):
            raise IndexError(f"Frame index {frame_idx} out of range")
        
        # Convert 3x4 to 4x4
        T = np.eye(4)
        T[:3, :] = self.poses[frame_idx]
        return T


class KITTIStereo2015Loader:
    """
    Loader for KITTI Stereo 2015 dataset (for depth evaluation).
    
    Dataset structure:
    data_scene_flow/
    ├── training/
    │   ├── image_2/  (left color images)
    │   ├── image_3/  (right color images)
    │   ├── disp_noc_0/  (disparity ground truth, non-occluded)
    │   ├── disp_occ_0/  (disparity ground truth, occluded)
    │   └── calib_cam_to_cam/
    └── testing/
    """
    
    def __init__(self, dataset_path: str, split: str = "training"):
        """
        Initialize KITTI Stereo 2015 loader.
        
        Args:
            dataset_path: Path to KITTI stereo 2015 dataset root
            split: 'training' or 'testing'
        """
        self.dataset_path = dataset_path
        self.split = split
        
        # Check if using standard structure or direct structure
        standard_split_path = os.path.join(dataset_path, split)
        
        if os.path.exists(standard_split_path):
            self.split_path = standard_split_path
        else:
            # Assume direct structure (images directly in dataset_path)
            self.split_path = dataset_path
        
        self.left_dir = os.path.join(self.split_path, "image_2")
        self.right_dir = os.path.join(self.split_path, "image_3")
        self.disp_noc_dir = os.path.join(self.split_path, "disp_noc_0")
        self.disp_occ_dir = os.path.join(self.split_path, "disp_occ_0")
        self.calib_dir = os.path.join(self.split_path, "calib_cam_to_cam")
        
        # Count number of images
        if os.path.exists(self.left_dir):
            self.num_images = len([f for f in os.listdir(self.left_dir) if f.endswith('.png')])
        else:
            self.num_images = 0
    
    @property
    def num_samples(self) -> int:
        """Return number of samples (alias for num_images)."""
        return self.num_images
    
    def get_stereo_pair(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load stereo image pair.
        
        Args:
            idx: Image index (e.g., 0 for 000000_10.png)
        
        Returns:
            Tuple of (left_image, right_image)
        """
        left_files = sorted([f for f in os.listdir(self.left_dir) if f.endswith('.png')])
        right_files = sorted([f for f in os.listdir(self.right_dir) if f.endswith('.png')])
        
        left_path = os.path.join(self.left_dir, left_files[idx])
        right_path = os.path.join(self.right_dir, right_files[idx])
        
        left_img = cv2.imread(left_path)
        right_img = cv2.imread(right_path)
        
        return left_img, right_img
    
    def get_image_pair(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load stereo image pair (alias for get_stereo_pair for compatibility).
        
        Args:
            idx: Image index
        
        Returns:
            Tuple of (left_image, right_image)
        """
        return self.get_stereo_pair(idx)
    
    def get_disparity_gt(self, idx: int, occluded: bool = False) -> np.ndarray:
        """
        Load ground truth disparity map.
        
        Args:
            idx: Image index
            occluded: If True, load occluded disparity; otherwise non-occluded
        
        Returns:
            Disparity map (already divided by 256.0)
        """
        disp_dir = self.disp_occ_dir if occluded else self.disp_noc_dir
        
        if not os.path.exists(disp_dir):
            raise FileNotFoundError(f"Disparity directory not found: {disp_dir}")
        
        disp_files = sorted([f for f in os.listdir(disp_dir) if f.endswith('.png')])
        disp_path = os.path.join(disp_dir, disp_files[idx])
        
        # KITTI disparity is saved as uint16 PNG, needs to be divided by 256
        disp = cv2.imread(disp_path, cv2.IMREAD_ANYDEPTH)
        disp = disp.astype(np.float32) / 256.0
        
        # 0 means invalid
        disp[disp == 0] = -1
        
        return disp
    
    def load_calibration(self, idx: int) -> Dict[str, np.ndarray]:
        """
        Load calibration for specific image.
        
        Args:
            idx: Image index
        
        Returns:
            Dictionary with calibration data
        """
        calib_files = sorted([f for f in os.listdir(self.calib_dir) if f.endswith('.txt')])
        calib_path = os.path.join(self.calib_dir, calib_files[idx])
        
        calib = {}
        with open(calib_path, 'r') as f:
            for line in f:
                if ':' in line:
                    key, value = line.split(':', 1)
                    calib[key.strip()] = value.strip()
        
        return calib


def disparity_to_depth(disparity: np.ndarray, focal_length: float, baseline: float) -> np.ndarray:
    """
    Convert disparity map to depth map using stereo geometry.
    
    Z = (f * B) / d
    where:
        Z = depth
        f = focal length
        B = baseline
        d = disparity
    
    Args:
        disparity: Disparity map
        focal_length: Camera focal length in pixels
        baseline: Stereo baseline in meters
    
    Returns:
        Depth map in meters
    """
    # Avoid division by zero
    depth = np.zeros_like(disparity)
    valid = disparity > 0
    depth[valid] = (focal_length * baseline) / disparity[valid]
    
    return depth
