"""
Visual Odometry Module
Implements classical visual odometry for camera motion estimation using:
1. Feature detection and matching
2. RANSAC-based geometry estimation (Essential/Fundamental matrix)
3. Pose recovery (R, t)
4. Scale recovery using stereo depth
"""

import numpy as np
import cv2
from typing import Tuple, Optional, List, Dict
from tqdm import tqdm


class VisualOdometry:
    """
    Classical stereo visual odometry system.
    
    Pipeline:
    1. Detect keypoints in left images (t and t+1)
    2. Match features between frames
    3. Use RANSAC to estimate Essential/Fundamental matrix
    4. Recover relative pose (R, t)
    5. Use stereo depth to recover metric scale
    6. Chain transformations to get full trajectory
    """
    
    def __init__(self,
                 K: np.ndarray,
                 baseline: float,
                 focal_length: float,
                 feature_detector: str = 'SIFT',
                 max_features: int = 3000,
                 ransac_threshold: float = 1.0,
                 ransac_confidence: float = 0.99):
        """
        Initialize Visual Odometry system.
        
        Args:
            K: Camera intrinsic matrix (3x3)
            baseline: Stereo baseline in meters
            focal_length: Focal length in pixels
            feature_detector: Type of feature detector ('ORB', 'SIFT', 'AKAZE')
            max_features: Maximum number of features to detect
            ransac_threshold: RANSAC inlier threshold
            ransac_confidence: RANSAC confidence level
        """
        self.K = K
        self.baseline = baseline
        self.focal_length = focal_length
        self.max_features = max_features
        self.ransac_threshold = ransac_threshold
        self.ransac_confidence = ransac_confidence
        
        # Initialize feature detector
        self.feature_detector = self._create_detector(feature_detector)
        
        # Initialize feature matcher
        if feature_detector == 'ORB' or feature_detector == 'AKAZE':
            # Binary descriptors use Hamming distance
            self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        else:
            # Float descriptors use L2 distance
            self.matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        
        # Trajectory storage
        self.poses = [np.eye(4)]  # Start at origin
    
    def _create_detector(self, detector_type: str):
        """
        Create feature detector.
        
        Args:
            detector_type: Type of detector
        
        Returns:
            Feature detector object
        """
        if detector_type == 'ORB':
            return cv2.ORB_create(nfeatures=self.max_features)
        elif detector_type == 'SIFT':
            return cv2.SIFT_create(nfeatures=self.max_features)
        elif detector_type == 'AKAZE':
            return cv2.AKAZE_create()
        else:
            raise ValueError(f"Unknown detector type: {detector_type}")
    
    def detect_and_compute(self, image: np.ndarray) -> Tuple[List, np.ndarray]:
        """
        Detect keypoints and compute descriptors.
        
        Args:
            image: Input image (grayscale)
        
        Returns:
            Tuple of (keypoints, descriptors)
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        keypoints, descriptors = self.feature_detector.detectAndCompute(image, None)
        
        return keypoints, descriptors
    
    def match_features(self,
                      desc1: np.ndarray,
                      desc2: np.ndarray,
                      ratio_threshold: float = 0.75) -> List[cv2.DMatch]:
        """
        Match features using ratio test (Lowe's ratio test).
        
        Args:
            desc1: Descriptors from first image
            desc2: Descriptors from second image
            ratio_threshold: Ratio threshold for Lowe's ratio test
        
        Returns:
            List of good matches
        """
        # Find 2 nearest neighbors
        matches = self.matcher.knnMatch(desc1, desc2, k=2)
        
        # Apply ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < ratio_threshold * n.distance:
                    good_matches.append(m)
        
        return good_matches
    
    def estimate_essential_matrix(self,
                                 pts1: np.ndarray,
                                 pts2: np.ndarray,
                                 use_ransac: bool = True) -> Tuple[np.ndarray, np.ndarray]:
        """
        Estimate Essential matrix with or without RANSAC.
        
        Args:
            pts1: Points in first image (Nx2)
            pts2: Points in second image (Nx2)
            use_ransac: If True use RANSAC (robust), else use 8-point algorithm (no outlier rejection)
        
        Returns:
            Tuple of (Essential matrix, inlier mask)
        """
        if use_ransac:
            E, mask = cv2.findEssentialMat(
                pts1, pts2, self.K,
                method=cv2.RANSAC,
                prob=self.ransac_confidence,
                threshold=self.ransac_threshold
            )
        else:
            # 8-point algorithm — no outlier rejection, all points treated as inliers
            E, mask = cv2.findEssentialMat(
                pts1, pts2, self.K,
                method=cv2.LMEDS  # least-median — weaker than RANSAC but still mild
            )
            # Mark all as inliers (no filtering)
            mask = np.ones((len(pts1), 1), dtype=np.uint8)
        
        return E, mask
    
    def estimate_fundamental_matrix(self,
                                   pts1: np.ndarray,
                                   pts2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Estimate Fundamental matrix using RANSAC.
        
        Args:
            pts1: Points in first image (Nx2)
            pts2: Points in second image (Nx2)
        
        Returns:
            Tuple of (Fundamental matrix, inlier mask)
        """
        F, mask = cv2.findFundamentalMat(
            pts1, pts2,
            method=cv2.FM_RANSAC,
            ransacReprojThreshold=self.ransac_threshold,
            confidence=self.ransac_confidence
        )
        
        return F, mask
    
    def recover_pose(self,
                    E: np.ndarray,
                    pts1: np.ndarray,
                    pts2: np.ndarray,
                    mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Recover relative pose from Essential matrix.
        
        Args:
            E: Essential matrix
            pts1: Points in first image (Nx2)
            pts2: Points in second image (Nx2)
            mask: Inlier mask from RANSAC
        
        Returns:
            Tuple of (R, t, mask_pose) where R is rotation, t is translation direction
        """
        # Recover pose from essential matrix
        _, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, self.K, mask=mask)
        
        return R, t, mask_pose
    
    def triangulate_points(self,
                          pts1: np.ndarray,
                          pts2: np.ndarray,
                          P1: np.ndarray,
                          P2: np.ndarray) -> np.ndarray:
        """
        Triangulate 3D points from 2D correspondences.
        
        Args:
            pts1: Points in first image (Nx2)
            pts2: Points in second image (Nx2)
            P1: Projection matrix for first camera (3x4)
            P2: Projection matrix for second camera (3x4)
        
        Returns:
            3D points (Nx3)
        """
        # Triangulate points
        pts_4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
        
        # Convert from homogeneous to 3D
        pts_3d = pts_4d[:3, :] / pts_4d[3, :]
        
        return pts_3d.T
    
    def compute_stereo_3d_points(self,
                                pts_left: np.ndarray,
                                disparity_map: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute 3D points from 2D points and disparity map.
        
        Args:
            pts_left: 2D points in left image (Nx2)
            disparity_map: Disparity map
        
        Returns:
            Tuple of (3D points, valid mask)
        """
        pts_3d = []
        valid_mask = []
        
        for pt in pts_left:
            x, y = int(pt[0]), int(pt[1])
            
            # Check if point is within image bounds
            if x < 0 or x >= disparity_map.shape[1] or y < 0 or y >= disparity_map.shape[0]:
                pts_3d.append([0, 0, 0])
                valid_mask.append(False)
                continue
            
            # Get disparity
            d = disparity_map[y, x]
            
            # Check if disparity is valid
            if d <= 0:
                pts_3d.append([0, 0, 0])
                valid_mask.append(False)
                continue
            
            # Compute 3D point
            Z = (self.focal_length * self.baseline) / d
            X = (x - self.K[0, 2]) * Z / self.K[0, 0]
            Y = (y - self.K[1, 2]) * Z / self.K[1, 1]
            
            pts_3d.append([X, Y, Z])
            valid_mask.append(True)
        
        return np.array(pts_3d), np.array(valid_mask)
    
    def estimate_pose_pnp(self,
                         pts_3d: np.ndarray,
                         pts_2d: np.ndarray,
                         use_ransac: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Estimate camera pose using PnP (Perspective-n-Point).
        
        Args:
            pts_3d: 3D points in world coordinates (Nx3)
            pts_2d: Corresponding 2D points in image (Nx2)
            use_ransac: Use RANSAC for robust estimation
        
        Returns:
            Tuple of (R, t, inliers) where R is rotation matrix and t is translation vector
        """
        if use_ransac:
            success, rvec, tvec, inliers = cv2.solvePnPRansac(
                pts_3d, pts_2d, self.K, None,
                reprojectionError=self.ransac_threshold,
                confidence=self.ransac_confidence
            )
        else:
            success, rvec, tvec = cv2.solvePnP(pts_3d, pts_2d, self.K, None)
            inliers = np.ones((len(pts_3d), 1), dtype=np.uint8)
        
        if not success:
            return None, None, None
        
        # Convert rotation vector to rotation matrix
        R, _ = cv2.Rodrigues(rvec)
        
        return R, tvec, inliers
    
    def process_frame(self,
                     left_img_prev: np.ndarray,
                     left_img_curr: np.ndarray,
                     disparity_map_prev: Optional[np.ndarray] = None,
                     method: str = 'pnp',
                     use_ransac: bool = True) -> Dict:
        """
        Process a frame pair to estimate relative motion.
        
        Args:
            left_img_prev: Previous left image
            left_img_curr: Current left image
            disparity_map_prev: Disparity map for previous frame (required for PnP)
            method: 'essential' or 'pnp' (pnp uses stereo depth for scale)
            use_ransac: If False, skips RANSAC outlier rejection (shows RANSAC effect)
        
        Returns:
            Dictionary containing motion estimate and inliers
        """
        # Detect and match features
        kp1, desc1 = self.detect_and_compute(left_img_prev)
        kp2, desc2 = self.detect_and_compute(left_img_curr)
        
        if desc1 is None or desc2 is None or len(kp1) < 8 or len(kp2) < 8:
            return {
                'success': False,
                'R': np.eye(3),
                't': np.zeros((3, 1)),
                'num_inliers': 0
            }
        
        # Match features
        matches = self.match_features(desc1, desc2)
        
        if len(matches) < 8:
            return {
                'success': False,
                'R': np.eye(3),
                't': np.zeros((3, 1)),
                'num_inliers': 0
            }
        
        # Extract matched points
        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])
        
        if method == 'pnp' and disparity_map_prev is not None:
            # Use PnP with stereo depth for metric scale
            pts_3d, valid_mask = self.compute_stereo_3d_points(pts1, disparity_map_prev)
            
            # Filter valid points
            pts_3d_valid = pts_3d[valid_mask]
            pts_2d_valid = pts2[valid_mask]
            
            if len(pts_3d_valid) < 8:
                # Fall back to essential matrix
                method = 'essential'
            else:
                # Estimate pose using PnP
                R, t, inliers = self.estimate_pose_pnp(pts_3d_valid, pts_2d_valid, use_ransac=use_ransac)
                
                if R is None:
                    return {
                        'success': False,
                        'R': np.eye(3),
                        't': np.zeros((3, 1)),
                        'num_inliers': 0
                    }
                
                return {
                    'success': True,
                    'R': R,
                    't': t,
                    'num_inliers': len(inliers),
                    'total_matches': len(matches),
                    'pts1': pts1,
                    'pts2': pts2,
                    'inliers': inliers
                }
        
        if method == 'essential' or method == 'pnp':
            # Estimate Essential matrix
            E, mask = self.estimate_essential_matrix(pts1, pts2, use_ransac=use_ransac)
            
            if E is None:
                return {
                    'success': False,
                    'R': np.eye(3),
                    't': np.zeros((3, 1)),
                    'num_inliers': 0
                }
            
            # Recover pose
            R, t, mask_pose = self.recover_pose(E, pts1, pts2, mask)
            
            # Note: scale is arbitrary without stereo depth
            return {
                'success': True,
                'R': R,
                't': t,
                'num_inliers': np.sum(mask),
                'total_matches': len(matches),
                'pts1': pts1,
                'pts2': pts2,
                'inliers': mask
            }
        
        return {
            'success': False,
            'R': np.eye(3),
            't': np.zeros((3, 1)),
            'num_inliers': 0
        }
    
    def update_pose(self, R: np.ndarray, t: np.ndarray) -> np.ndarray:
        """
        Update current pose with relative transformation.
        
        Args:
            R: Relative rotation matrix
            t: Relative translation vector
        
        Returns:
            Updated absolute pose (4x4)
        """
        # Get current pose
        T_curr = self.poses[-1]
        
        R_inv = R.T
        t_inv = -R_inv @ t

        # Create relative transformation
        T_rel_inv = np.eye(4)
        T_rel_inv[:3, :3] = R_inv
        T_rel_inv[:3, 3:4] = t_inv
        
        # Update pose: T_new = T_curr @ T_rel
        T_new = T_curr @ T_rel_inv
        
        # Store pose
        self.poses.append(T_new)
        
        return T_new
    
    def get_trajectory(self) -> np.ndarray:
        """
        Get full trajectory as array of poses.
        
        Returns:
            Array of poses (Nx4x4)
        """
        return np.array(self.poses)
    
    def reset(self):
        """Reset trajectory to initial state."""
        self.poses = [np.eye(4)]
