"""
Visualization utilities for stereo depth and visual odometry.
Provides functions to visualize disparity maps, depth maps, feature matches,
and camera trajectories.
"""

import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib import cm
from typing import Tuple, List, Optional
import os


def visualize_disparity(disparity: np.ndarray, 
                       title: str = "Disparity Map",
                       save_path: Optional[str] = None,
                       vmin: float = 0,
                       vmax: Optional[float] = None) -> None:
    """
    Visualize disparity map with colormap.
    
    Args:
        disparity: Disparity map
        title: Plot title
        save_path: Path to save figure (optional)
        vmin: Minimum disparity value for colormap
        vmax: Maximum disparity value for colormap
    """
    plt.figure(figsize=(12, 6))
    
    # Mask invalid disparities
    valid_mask = disparity > 0
    disp_vis = np.copy(disparity)
    disp_vis[~valid_mask] = np.nan
    
    if vmax is None:
        vmax = np.nanpercentile(disp_vis, 95)
    
    plt.imshow(disp_vis, cmap='jet', vmin=vmin, vmax=vmax)
    plt.colorbar(label='Disparity (pixels)')
    plt.title(title)
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Saved disparity visualization to {save_path}")
    
    plt.close()  # Close figure to free memory


def visualize_depth(depth: np.ndarray,
                   title: str = "Depth Map",
                   save_path: Optional[str] = None,
                   max_depth: float = 80.0) -> None:
    """
    Visualize depth map with colormap.
    
    Args:
        depth: Depth map in meters
        title: Plot title
        save_path: Path to save figure (optional)
        max_depth: Maximum depth for visualization (meters)
    """
    plt.figure(figsize=(12, 6))
    
    # Mask invalid depths
    valid_mask = depth > 0
    depth_vis = np.copy(depth)
    depth_vis[~valid_mask] = np.nan
    depth_vis = np.clip(depth_vis, 0, max_depth)
    
    plt.imshow(depth_vis, cmap='plasma_r', vmin=0, vmax=max_depth)
    plt.colorbar(label='Depth (meters)')
    plt.title(title)
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Saved depth visualization to {save_path}")
    
    plt.close()


def visualize_matches(img1: np.ndarray,
                     img2: np.ndarray,
                     pts1: np.ndarray,
                     pts2: np.ndarray,
                     inliers: Optional[np.ndarray] = None,
                     title: str = "Feature Matches",
                     save_path: Optional[str] = None,
                     max_matches: int = 100) -> None:
    """
    Visualize feature matches between two images.
    
    Args:
        img1: First image
        img2: Second image
        pts1: Points in first image (Nx2)
        pts2: Points in second image (Nx2)
        inliers: Boolean mask indicating inliers (optional)
        title: Plot title
        save_path: Path to save figure (optional)
        max_matches: Maximum number of matches to display
    """
    # Convert grayscale to color if needed
    if len(img1.shape) == 2:
        img1_color = cv2.cvtColor(img1, cv2.COLOR_GRAY2RGB)
    else:
        img1_color = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    
    if len(img2.shape) == 2:
        img2_color = cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB)
    else:
        img2_color = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    
    # Concatenate images side by side
    h1, w1 = img1_color.shape[:2]
    h2, w2 = img2_color.shape[:2]
    h = max(h1, h2)
    
    canvas = np.zeros((h, w1 + w2, 3), dtype=np.uint8)
    canvas[:h1, :w1] = img1_color
    canvas[:h2, w1:w1+w2] = img2_color
    
    # Subsample matches if too many
    original_length = len(pts1)
    if original_length > max_matches:
        indices = np.random.choice(original_length, max_matches, replace=False)
        pts1 = pts1[indices]
        pts2 = pts2[indices]
        # Only subsample inliers if it matches the original length
        if inliers is not None:
            if len(inliers) == original_length:
                inliers = inliers[indices]
            else:
                # Inliers array length doesn't match - cannot safely subsample
                print(f"Warning: inliers length ({len(inliers)}) != pts1 length ({original_length}). Cannot subsample inliers.")
                inliers = None
    
    # Draw matches
    for i in range(len(pts1)):
        pt1 = tuple(pts1[i].astype(int))
        pt2 = tuple((pts2[i] + [w1, 0]).astype(int))
        
        if inliers is not None:
            color = (0, 255, 0) if inliers[i] else (255, 0, 0)
            thickness = 2 if inliers[i] else 1
        else:
            color = (0, 255, 0)
            thickness = 1
        
        cv2.line(canvas, pt1, pt2, color, thickness)
        cv2.circle(canvas, pt1, 3, color, -1)
        cv2.circle(canvas, pt2, 3, color, -1)
    
    plt.figure(figsize=(16, 8))
    plt.imshow(canvas)
    plt.title(title)
    plt.axis('off')
    
    if inliers is not None:
        n_inliers = np.sum(inliers)
        plt.text(10, 30, f"Inliers: {n_inliers}/{len(pts1)}", 
                color='green', fontsize=12, weight='bold',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Saved matches visualization to {save_path}")
    
    plt.close()


def visualize_trajectory(estimated_poses: np.ndarray,
                        gt_poses: Optional[np.ndarray] = None,
                        title: str = "Camera Trajectory",
                        save_path: Optional[str] = None) -> None:
    """
    Visualize camera trajectory in bird's eye view (XZ plane).
    
    Args:
        estimated_poses: Estimated camera poses (Nx4x4 or Nx3x4)
        gt_poses: Ground truth poses (optional, Nx4x4 or Nx3x4)
        title: Plot title
        save_path: Path to save figure (optional)
    """
    fig, ax = plt.subplots(figsize=(12, 12))
    
    # Extract translations from estimated poses
    if estimated_poses.shape[1:] == (4, 4):
        est_x = estimated_poses[:, 0, 3]
        est_z = estimated_poses[:, 2, 3]
    else:  # (3, 4)
        est_x = estimated_poses[:, 0, 3]
        est_z = estimated_poses[:, 2, 3]
    
    # Plot estimated trajectory
    ax.plot(est_x, est_z, 'b-', label='Estimated', linewidth=2, alpha=0.7)
    ax.plot(est_x[0], est_z[0], 'go', markersize=10, label='Start')
    ax.plot(est_x[-1], est_z[-1], 'ro', markersize=10, label='End')
    
    # Plot ground truth if available
    if gt_poses is not None:
        if gt_poses.shape[1:] == (4, 4):
            gt_x = gt_poses[:, 0, 3]
            gt_z = gt_poses[:, 2, 3]
        else:  # (3, 4)
            gt_x = gt_poses[:, 0, 3]
            gt_z = gt_poses[:, 2, 3]
        
        ax.plot(gt_x, gt_z, 'k--', label='Ground Truth', linewidth=2, alpha=0.5)
    
    ax.set_xlabel('X (meters)', fontsize=12)
    ax.set_ylabel('Z (meters)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Saved trajectory visualization to {save_path}")
    
    plt.close()


def create_disparity_comparison(left_img: np.ndarray,
                                disparity_maps: List[Tuple[np.ndarray, str]],
                                save_path: Optional[str] = None) -> None:
    """
    Create a comparison figure showing left image and multiple disparity maps.
    
    Args:
        left_img: Left stereo image
        disparity_maps: List of (disparity_map, method_name) tuples
        save_path: Path to save figure (optional)
    """
    n_maps = len(disparity_maps)
    fig, axes = plt.subplots(1, n_maps + 1, figsize=(6 * (n_maps + 1), 6))
    
    # Show left image
    if len(left_img.shape) == 2:
        axes[0].imshow(left_img, cmap='gray')
    else:
        axes[0].imshow(cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Left Image')
    axes[0].axis('off')
    
    # Show disparity maps
    for i, (disp, method) in enumerate(disparity_maps):
        valid_mask = disp > 0
        disp_vis = np.copy(disp)
        disp_vis[~valid_mask] = np.nan
        vmax = np.nanpercentile(disp_vis, 95)
        
        im = axes[i + 1].imshow(disp_vis, cmap='jet', vmin=0, vmax=vmax)
        axes[i + 1].set_title(f'{method}')
        axes[i + 1].axis('off')
        plt.colorbar(im, ax=axes[i + 1], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"Saved comparison to {save_path}")
    
    plt.close()


def save_disparity_png(disparity: np.ndarray, save_path: str) -> None:
    """
    Save disparity map as 16-bit PNG (KITTI format).
    
    Args:
        disparity: Disparity map (in pixels)
        save_path: Path to save PNG file
    """
    # Convert to uint16 (multiply by 256 for KITTI format)
    disp_uint16 = (disparity * 256.0).astype(np.uint16)
    cv2.imwrite(save_path, disp_uint16)
    print(f"Saved disparity to {save_path}")


def visualize_error_map(predicted: np.ndarray,
                       ground_truth: np.ndarray,
                       title: str = "Error Map",
                       save_path: Optional[str] = None,
                       max_error: float = 10.0) -> None:
    """
    Visualize error between predicted and ground truth disparity/depth.
    
    Args:
        predicted: Predicted disparity or depth
        ground_truth: Ground truth disparity or depth
        title: Plot title
        save_path: Path to save figure (optional)
        max_error: Maximum error for colormap
    """
    # Compute absolute error
    valid_mask = (ground_truth > 0) & (predicted > 0)
    error = np.abs(predicted - ground_truth)
    error[~valid_mask] = np.nan
    
    plt.figure(figsize=(12, 6))
    plt.imshow(error, cmap='hot', vmin=0, vmax=max_error)
    plt.colorbar(label='Absolute Error')
    plt.title(title)
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    
    plt.close()


def plot_error_histogram(errors: np.ndarray,
                         title: str = "Error Distribution",
                         save_path: Optional[str] = None) -> None:
    """
    Plot histogram of errors.
    
    Args:
        errors: Array of errors
        title: Plot title
        save_path: Path to save figure (optional)
    """
    plt.figure(figsize=(10, 6))
    plt.hist(errors[~np.isnan(errors)], bins=50, edgecolor='black', alpha=0.7)
    plt.xlabel('Error')
    plt.ylabel('Frequency')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    
    # Add statistics
    mean_error = np.nanmean(errors)
    median_error = np.nanmedian(errors)
    plt.axvline(mean_error, color='r', linestyle='--', label=f'Mean: {mean_error:.2f}')
    plt.axvline(median_error, color='g', linestyle='--', label=f'Median: {median_error:.2f}')
    plt.legend()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    
    plt.close()


def visualize_disparity_comparison(predicted: np.ndarray,
                                   ground_truth: np.ndarray,
                                   left_img: Optional[np.ndarray] = None,
                                   title: str = "Disparity Comparison",
                                   max_error: float = 5.0,
                                   focal_length: float = None,
                                   baseline: float = None) -> plt.Figure:
    """
    Create comprehensive comparison visualization of predicted vs ground truth disparity.
    Shows left image (if provided), predicted disparity, ground truth disparity, error map,
    and optionally depth map (if focal_length and baseline are provided).
    
    Args:
        predicted: Predicted disparity map
        ground_truth: Ground truth disparity map
        left_img: Left stereo image (optional)
        title: Main title for the figure
        max_error: Maximum error for error map colormap
        focal_length: Camera focal length in pixels (for depth computation)
        baseline: Stereo baseline in meters (for depth computation)
        
    Returns:
        matplotlib Figure object
    """
    # Determine number of subplots
    show_depth = focal_length is not None and baseline is not None
    n_plots = (4 if left_img is not None else 3) + (1 if show_depth else 0)
    
    # Create figure
    fig, axes = plt.subplots(1, n_plots, figsize=(6 * n_plots, 6))
    
    plot_idx = 0
    
    # Show left image if provided
    if left_img is not None:
        if len(left_img.shape) == 2:
            axes[plot_idx].imshow(left_img, cmap='gray')
        else:
            axes[plot_idx].imshow(cv2.cvtColor(left_img, cv2.COLOR_BGR2RGB))
        axes[plot_idx].set_title('Left Image', fontsize=14)
        axes[plot_idx].axis('off')
        plot_idx += 1
    
    # Compute valid mask
    valid_mask = ground_truth > 0
    
    # Show predicted disparity
    pred_vis = np.copy(predicted)
    pred_vis[~valid_mask] = np.nan
    vmax_pred = np.nanpercentile(pred_vis, 95)
    
    im1 = axes[plot_idx].imshow(pred_vis, cmap='jet', vmin=0, vmax=vmax_pred)
    axes[plot_idx].set_title('Predicted Disparity', fontsize=14)
    axes[plot_idx].axis('off')
    plt.colorbar(im1, ax=axes[plot_idx], fraction=0.046, pad=0.04, label='Disparity (pixels)')
    plot_idx += 1
    
    # Show ground truth disparity
    gt_vis = np.copy(ground_truth)
    gt_vis[~valid_mask] = np.nan
    vmax_gt = np.nanpercentile(gt_vis, 95)
    
    im2 = axes[plot_idx].imshow(gt_vis, cmap='jet', vmin=0, vmax=vmax_gt)
    axes[plot_idx].set_title('Ground Truth Disparity', fontsize=14)
    axes[plot_idx].axis('off')
    plt.colorbar(im2, ax=axes[plot_idx], fraction=0.046, pad=0.04, label='Disparity (pixels)')
    plot_idx += 1
    
    # Show error map
    error = np.abs(predicted - ground_truth)
    error[~valid_mask] = np.nan
    
    im3 = axes[plot_idx].imshow(error, cmap='hot', vmin=0, vmax=max_error)
    axes[plot_idx].set_title('Absolute Error', fontsize=14)
    axes[plot_idx].axis('off')
    cbar = plt.colorbar(im3, ax=axes[plot_idx], fraction=0.046, pad=0.04, label='Error (pixels)')
    
    # Add error statistics as text
    mae = np.nanmean(error)
    rmse = np.sqrt(np.nanmean(error ** 2))
    bad_pixels = np.sum(error[~np.isnan(error)] > 3.0) / np.sum(~np.isnan(error)) * 100
    
    stats_text = f'MAE: {mae:.2f}px\nRMSE: {rmse:.2f}px\nBad-3: {bad_pixels:.1f}%'
    axes[plot_idx].text(0.02, 0.98, stats_text, transform=axes[plot_idx].transAxes,
                       fontsize=11, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    plot_idx += 1

    # Show depth map if calibration provided
    if show_depth:
        depth = np.zeros_like(predicted)
        valid_d = predicted > 0
        depth[valid_d] = (focal_length * baseline) / predicted[valid_d]
        depth[~valid_mask] = np.nan
        depth_clipped = np.clip(depth, 0, 80.0)

        im4 = axes[plot_idx].imshow(depth_clipped, cmap='plasma_r', vmin=0, vmax=80.0)
        axes[plot_idx].set_title('Depth Map (meters)', fontsize=14)
        axes[plot_idx].axis('off')
        plt.colorbar(im4, ax=axes[plot_idx], fraction=0.046, pad=0.04, label='Depth (m)')

        valid_depth = depth[~np.isnan(depth)]
        if len(valid_depth) > 0:
            depth_text = f'Mean: {np.mean(valid_depth):.1f}m\nMedian: {np.median(valid_depth):.1f}m\nMax: {np.percentile(valid_depth, 95):.1f}m'
            axes[plot_idx].text(0.02, 0.98, depth_text, transform=axes[plot_idx].transAxes,
                               fontsize=11, verticalalignment='top',
                               bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Set main title
    fig.suptitle(title, fontsize=16, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    
    return fig
