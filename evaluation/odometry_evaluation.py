"""
Visual Odometry Evaluation Script
Evaluates camera trajectory estimation on KITTI Odometry dataset.
Computes metrics: ATE (Absolute Trajectory Error), RPE (Relative Pose Error)
"""

import numpy as np
import sys
import os
from typing import Dict, Tuple, List

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def align_trajectories(estimated: np.ndarray, ground_truth: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Align estimated trajectory to ground truth using Umeyama alignment (sim3).
    
    This computes the optimal similarity transformation (rotation, translation, scale)
    that aligns the estimated trajectory to ground truth.
    
    Args:
        estimated: Estimated trajectory (Nx4x4 or Nx3x4)
        ground_truth: Ground truth trajectory (Nx4x4 or Nx3x4)
    
    Returns:
        Tuple of (aligned_trajectory, scale_factor)
    """
    # Extract translation components
    if estimated.shape[1:] == (4, 4):
        est_xyz = estimated[:, :3, 3]
        gt_xyz = ground_truth[:, :3, 3]
    else:  # (3, 4)
        est_xyz = estimated[:, :3, 3]
        gt_xyz = ground_truth[:, :3, 3]
    
    # Compute centroids
    est_centroid = np.mean(est_xyz, axis=0)
    gt_centroid = np.mean(gt_xyz, axis=0)
    
    # Center the point clouds
    est_centered = est_xyz - est_centroid
    gt_centered = gt_xyz - gt_centroid
    
    # Compute scale
    est_scale = np.sqrt(np.sum(est_centered ** 2) / len(est_centered))
    gt_scale = np.sqrt(np.sum(gt_centered ** 2) / len(gt_centered))
    scale = gt_scale / est_scale
    
    # Scale estimated trajectory
    est_scaled = est_centered * scale
    
    # Compute rotation using SVD
    H = est_scaled.T @ gt_centered
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    
    # Ensure proper rotation (det(R) = 1)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    # Compute translation
    t = gt_centroid - scale * (R @ est_centroid)
    
    # Apply transformation to estimated trajectory
    aligned = np.copy(estimated)
    for i in range(len(estimated)):
        if estimated.shape[1:] == (4, 4):
            aligned[i, :3, 3] = scale * (R @ estimated[i, :3, 3]) + t
        else:
            aligned[i, :3, 3] = scale * (R @ estimated[i, :3, 3]) + t
    
    return aligned, scale


def compute_ate(estimated: np.ndarray, ground_truth: np.ndarray, align: bool = True) -> Dict:
    """
    Compute Absolute Trajectory Error (ATE).
    
    ATE measures the global consistency of the trajectory.
    
    Args:
        estimated: Estimated trajectory (Nx4x4 or Nx3x4)
        ground_truth: Ground truth trajectory (Nx4x4 or Nx3x4)
        align: Whether to align trajectories before computing error
    
    Returns:
        Dictionary with ATE statistics
    """
    # Ensure same length
    n = min(len(estimated), len(ground_truth))
    estimated = estimated[:n]
    ground_truth = ground_truth[:n]
    
    # Align if requested
    if align:
        estimated_aligned, scale = align_trajectories(estimated, ground_truth)
    else:
        estimated_aligned = estimated
        scale = 1.0
    
    # Extract translations
    if estimated_aligned.shape[1:] == (4, 4):
        est_xyz = estimated_aligned[:, :3, 3]
        gt_xyz = ground_truth[:, :3, 3]
    else:
        est_xyz = estimated_aligned[:, :3, 3]
        gt_xyz = ground_truth[:, :3, 3]
    
    # Compute errors
    errors = np.linalg.norm(est_xyz - gt_xyz, axis=1)
    
    return {
        'mean': np.mean(errors),
        'median': np.median(errors),
        'std': np.std(errors),
        'min': np.min(errors),
        'max': np.max(errors),
        'rmse': np.sqrt(np.mean(errors ** 2)),
        'scale': scale,
        'errors': errors
    }


def compute_rpe(estimated: np.ndarray, 
                ground_truth: np.ndarray,
                step: int = 1) -> Dict:
    """
    Compute Relative Pose Error (RPE).
    
    RPE measures the local consistency of the trajectory by looking at
    pose differences over a fixed time interval.
    
    Args:
        estimated: Estimated trajectory (Nx4x4)
        ground_truth: Ground truth trajectory (Nx4x4)
        step: Step size (number of frames between poses)
    
    Returns:
        Dictionary with RPE statistics for translation and rotation
    """
    # Ensure same length and 4x4 format
    n = min(len(estimated), len(ground_truth))
    estimated = estimated[:n]
    ground_truth = ground_truth[:n]
    
    # Convert to 4x4 if needed
    est_poses = np.zeros((n, 4, 4))
    gt_poses = np.zeros((n, 4, 4))
    
    for i in range(n):
        if estimated.shape[1:] == (4, 4):
            est_poses[i] = estimated[i]
            gt_poses[i] = ground_truth[i]
        else:  # (3, 4)
            est_poses[i, :3, :] = estimated[i]
            est_poses[i, 3, 3] = 1
            gt_poses[i, :3, :] = ground_truth[i]
            gt_poses[i, 3, 3] = 1
    
    trans_errors = []
    rot_errors = []
    
    # Compute relative pose errors
    for i in range(0, n - step):
        # Ground truth relative pose
        gt_rel = np.linalg.inv(gt_poses[i]) @ gt_poses[i + step]
        
        # Estimated relative pose
        est_rel = np.linalg.inv(est_poses[i]) @ est_poses[i + step]
        
        # Error in relative pose
        error = np.linalg.inv(gt_rel) @ est_rel
        
        # Translation error
        trans_error = np.linalg.norm(error[:3, 3])
        trans_errors.append(trans_error)
        
        # Rotation error (angle of rotation matrix)
        # trace(R) = 1 + 2*cos(angle)
        trace = np.trace(error[:3, :3])
        cos_angle = (trace - 1) / 2
        cos_angle = np.clip(cos_angle, -1, 1)  # Numerical stability
        rot_angle = np.arccos(cos_angle)
        rot_errors.append(np.degrees(rot_angle))
    
    trans_errors = np.array(trans_errors)
    rot_errors = np.array(rot_errors)
    
    return {
        'translation': {
            'mean': np.mean(trans_errors),
            'median': np.median(trans_errors),
            'std': np.std(trans_errors),
            'min': np.min(trans_errors),
            'max': np.max(trans_errors),
            'rmse': np.sqrt(np.mean(trans_errors ** 2)),
            'errors': trans_errors
        },
        'rotation': {
            'mean': np.mean(rot_errors),
            'median': np.median(rot_errors),
            'std': np.std(rot_errors),
            'min': np.min(rot_errors),
            'max': np.max(rot_errors),
            'rmse': np.sqrt(np.mean(rot_errors ** 2)),
            'errors': rot_errors
        },
        'step': step
    }


def compute_trajectory_length(poses: np.ndarray) -> float:
    """
    Compute total trajectory length.
    
    Args:
        poses: Trajectory (Nx4x4 or Nx3x4)
    
    Returns:
        Total length in meters
    """
    # Extract translations
    if poses.shape[1:] == (4, 4):
        xyz = poses[:, :3, 3]
    else:
        xyz = poses[:, :3, 3]
    
    # Compute distances between consecutive poses
    distances = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
    total_length = np.sum(distances)
    
    return total_length


def evaluate_odometry(estimated: np.ndarray,
                     ground_truth: np.ndarray,
                     rpe_steps: List[int] = [1, 5, 10]) -> Dict:
    """
    Comprehensive visual odometry evaluation.
    
    Args:
        estimated: Estimated trajectory (Nx4x4 or Nx3x4)
        ground_truth: Ground truth trajectory (Nx4x4 or Nx3x4)
        rpe_steps: List of step sizes for RPE computation
    
    Returns:
        Dictionary with all metrics
    """
    results = {}
    
    # ATE
    results['ate'] = compute_ate(estimated, ground_truth, align=True)
    
    # RPE at different steps
    results['rpe'] = {}
    for step in rpe_steps:
        if step < len(estimated):
            results['rpe'][f'step_{step}'] = compute_rpe(estimated, ground_truth, step)
    
    # Trajectory lengths
    results['length'] = {
        'estimated': compute_trajectory_length(estimated),
        'ground_truth': compute_trajectory_length(ground_truth)
    }
    
    # Number of poses
    results['num_poses'] = min(len(estimated), len(ground_truth))
    
    return results


def print_evaluation_results(results: Dict, sequence_name: str = "Sequence"):
    """
    Print evaluation results in a formatted way.
    
    Args:
        results: Dictionary with evaluation results
        sequence_name: Name of the sequence being evaluated
    """
    print(f"\n{'='*70}")
    print(f"Visual Odometry Evaluation: {sequence_name}")
    print(f"{'='*70}")
    
    print(f"\nAbsolute Trajectory Error (ATE):")
    ate = results['ate']
    print(f"  RMSE:   {ate['rmse']:.3f} m")
    print(f"  Mean:   {ate['mean']:.3f} m")
    print(f"  Median: {ate['median']:.3f} m")
    print(f"  Std:    {ate['std']:.3f} m")
    print(f"  Min:    {ate['min']:.3f} m")
    print(f"  Max:    {ate['max']:.3f} m")
    print(f"  Scale:  {ate['scale']:.4f}")
    
    print(f"\nRelative Pose Error (RPE):")
    for step_name, rpe in results['rpe'].items():
        step = rpe['step']
        print(f"  Step size: {step} frame(s)")
        print(f"    Translation:")
        print(f"      RMSE:   {rpe['translation']['rmse']:.4f} m")
        print(f"      Mean:   {rpe['translation']['mean']:.4f} m")
        print(f"      Median: {rpe['translation']['median']:.4f} m")
        print(f"    Rotation:")
        print(f"      RMSE:   {rpe['rotation']['rmse']:.4f} deg")
        print(f"      Mean:   {rpe['rotation']['mean']:.4f} deg")
        print(f"      Median: {rpe['rotation']['median']:.4f} deg")
    
    print(f"\nTrajectory Statistics:")
    print(f"  Number of poses: {results['num_poses']}")
    print(f"  Estimated length: {results['length']['estimated']:.2f} m")
    print(f"  Ground truth length: {results['length']['ground_truth']:.2f} m")
    length_error = abs(results['length']['estimated'] - results['length']['ground_truth'])
    length_error_pct = (length_error / results['length']['ground_truth']) * 100
    print(f"  Length error: {length_error:.2f} m ({length_error_pct:.2f}%)")
    
    print(f"{'='*70}\n")


def plot_ate_errors(ate_results: Dict, save_path: str = None):
    """
    Plot ATE errors over trajectory.
    
    Args:
        ate_results: ATE results dictionary from compute_ate
        save_path: Path to save plot
    """
    import matplotlib.pyplot as plt
    
    errors = ate_results['errors']
    frames = np.arange(len(errors))
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    
    # Plot errors
    ax.plot(frames, errors, 'b-', linewidth=1, label='ATE per frame')
    ax.axhline(ate_results['mean'], color='r', linestyle='--', 
               linewidth=2, label=f"Mean: {ate_results['mean']:.3f} m")
    ax.axhline(ate_results['rmse'], color='orange', linestyle='--',
               linewidth=2, label=f"RMSE: {ate_results['rmse']:.3f} m")
    
    ax.set_xlabel('Frame Index', fontsize=12)
    ax.set_ylabel('Absolute Trajectory Error (m)', fontsize=12)
    ax.set_title('ATE Over Trajectory', fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_rpe_errors(rpe_results: Dict, save_path: str = None):
    """
    Plot RPE translation and rotation errors.
    
    Args:
        rpe_results: RPE results dictionary from compute_rpe
        save_path: Path to save plot
    """
    import matplotlib.pyplot as plt
    
    trans_errors = rpe_results['translation']['errors']
    rot_errors = rpe_results['rotation']['errors']
    step = rpe_results['step']
    frames = np.arange(len(trans_errors))
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Translation errors
    ax1.plot(frames, trans_errors, 'b-', linewidth=1, alpha=0.7)
    ax1.axhline(rpe_results['translation']['mean'], color='r', linestyle='--',
                linewidth=2, label=f"Mean: {rpe_results['translation']['mean']:.4f} m")
    ax1.axhline(rpe_results['translation']['rmse'], color='orange', linestyle='--',
                linewidth=2, label=f"RMSE: {rpe_results['translation']['rmse']:.4f} m")
    ax1.set_xlabel('Frame Index', fontsize=11)
    ax1.set_ylabel('Translation Error (m)', fontsize=11)
    ax1.set_title(f'RPE Translation Error (Step={step} frames)', fontsize=13, fontweight='bold')
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Rotation errors
    ax2.plot(frames, rot_errors, 'g-', linewidth=1, alpha=0.7)
    ax2.axhline(rpe_results['rotation']['mean'], color='r', linestyle='--',
                linewidth=2, label=f"Mean: {rpe_results['rotation']['mean']:.4f} deg")
    ax2.axhline(rpe_results['rotation']['rmse'], color='orange', linestyle='--',
                linewidth=2, label=f"RMSE: {rpe_results['rotation']['rmse']:.4f} deg")
    ax2.set_xlabel('Frame Index', fontsize=11)
    ax2.set_ylabel('Rotation Error (degrees)', fontsize=11)
    ax2.set_title(f'RPE Rotation Error (Step={step} frames)', fontsize=13, fontweight='bold')
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_trajectory_comparison(estimated: np.ndarray, 
                              ground_truth: np.ndarray,
                              ate_results: Dict,
                              save_path: str = None):
    """
    Plot detailed trajectory comparison with multiple views.
    
    Args:
        estimated: Estimated trajectory (Nx4x4)
        ground_truth: Ground truth trajectory (Nx4x4)
        ate_results: ATE results for error coloring
        save_path: Path to save plot
    """
    import matplotlib.pyplot as plt
    from matplotlib import cm
    
    # Extract positions
    est_pos = estimated[:, :3, 3]
    gt_pos = ground_truth[:, :3, 3]
    
    # Align estimated to ground truth for visualization
    from evaluation.odometry_evaluation import align_trajectories
    aligned_est, scale = align_trajectories(estimated, ground_truth)
    est_pos_aligned = aligned_est[:, :3, 3]
    
    errors = ate_results['errors']
    
    fig = plt.figure(figsize=(16, 10))
    
    # Top view (X-Z plane)
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(gt_pos[:, 0], gt_pos[:, 2], 'g-', linewidth=2, label='Ground Truth', alpha=0.8)
    ax1.plot(est_pos_aligned[:, 0], est_pos_aligned[:, 2], 'r--', linewidth=2, label='Estimated', alpha=0.8)
    ax1.scatter(gt_pos[0, 0], gt_pos[0, 2], c='blue', s=100, marker='o', label='Start', zorder=5)
    ax1.scatter(gt_pos[-1, 0], gt_pos[-1, 2], c='black', s=100, marker='s', label='End', zorder=5)
    ax1.set_xlabel('X (m)', fontsize=11)
    ax1.set_ylabel('Z (m)', fontsize=11)
    ax1.set_title('Top View (X-Z)', fontsize=12, fontweight='bold')
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.axis('equal')
    
    # Side view (Y-Z plane)
    ax2 = plt.subplot(2, 3, 2)
    ax2.plot(gt_pos[:, 1], gt_pos[:, 2], 'g-', linewidth=2, label='Ground Truth', alpha=0.8)
    ax2.plot(est_pos_aligned[:, 1], est_pos_aligned[:, 2], 'r--', linewidth=2, label='Estimated', alpha=0.8)
    ax2.set_xlabel('Y (m)', fontsize=11)
    ax2.set_ylabel('Z (m)', fontsize=11)
    ax2.set_title('Side View (Y-Z)', fontsize=12, fontweight='bold')
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3)
    ax2.axis('equal')
    
    # Front view (X-Y plane)
    ax3 = plt.subplot(2, 3, 3)
    ax3.plot(gt_pos[:, 0], gt_pos[:, 1], 'g-', linewidth=2, label='Ground Truth', alpha=0.8)
    ax3.plot(est_pos_aligned[:, 0], est_pos_aligned[:, 1], 'r--', linewidth=2, label='Estimated', alpha=0.8)
    ax3.set_xlabel('X (m)', fontsize=11)
    ax3.set_ylabel('Y (m)', fontsize=11)
    ax3.set_title('Front View (X-Y)', fontsize=12, fontweight='bold')
    ax3.legend(loc='best', fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.axis('equal')
    
    # Error heatmap on trajectory
    ax4 = plt.subplot(2, 3, 4)
    scatter = ax4.scatter(est_pos_aligned[:, 0], est_pos_aligned[:, 2], 
                         c=errors, cmap='hot', s=30, alpha=0.8)
    ax4.plot(gt_pos[:, 0], gt_pos[:, 2], 'g-', linewidth=1, alpha=0.3, label='Ground Truth')
    cbar = plt.colorbar(scatter, ax=ax4)
    cbar.set_label('ATE Error (m)', fontsize=10)
    ax4.set_xlabel('X (m)', fontsize=11)
    ax4.set_ylabel('Z (m)', fontsize=11)
    ax4.set_title('Error Heatmap (Top View)', fontsize=12, fontweight='bold')
    ax4.legend(loc='best', fontsize=9)
    ax4.grid(True, alpha=0.3)
    ax4.axis('equal')
    
    # Per-axis position errors
    ax5 = plt.subplot(2, 3, 5)
    frames = np.arange(len(est_pos_aligned))
    error_x = np.abs(est_pos_aligned[:, 0] - gt_pos[:, 0])
    error_y = np.abs(est_pos_aligned[:, 1] - gt_pos[:, 1])
    error_z = np.abs(est_pos_aligned[:, 2] - gt_pos[:, 2])
    ax5.plot(frames, error_x, 'r-', label='X error', alpha=0.7)
    ax5.plot(frames, error_y, 'g-', label='Y error', alpha=0.7)
    ax5.plot(frames, error_z, 'b-', label='Z error', alpha=0.7)
    ax5.set_xlabel('Frame Index', fontsize=11)
    ax5.set_ylabel('Position Error (m)', fontsize=11)
    ax5.set_title('Per-Axis Position Errors', fontsize=12, fontweight='bold')
    ax5.legend(loc='best', fontsize=9)
    ax5.grid(True, alpha=0.3)
    
    # Statistics box
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    stats_text = f"""
    TRAJECTORY STATISTICS
    ═══════════════════════════
    
    Number of Poses: {len(estimated)}
    
    ATE (Absolute Trajectory Error):
      RMSE:   {ate_results['rmse']:.3f} m
      Mean:   {ate_results['mean']:.3f} m
      Median: {ate_results['median']:.3f} m
      Std:    {ate_results['std']:.3f} m
      Min:    {ate_results['min']:.3f} m
      Max:    {ate_results['max']:.3f} m
    
    Scale Factor: {ate_results['scale']:.4f}
    
    Per-Axis Mean Errors:
      X: {np.mean(error_x):.3f} m
      Y: {np.mean(error_y):.3f} m
      Z: {np.mean(error_z):.3f} m
    """
    
    ax6.text(0.1, 0.95, stats_text, transform=ax6.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))
    
    plt.suptitle('Detailed Trajectory Comparison', fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def generate_evaluation_report(results: Dict, output_path: str, sequence_name: str = "Sequence"):
    """
    Generate comprehensive text report of evaluation results.
    
    Args:
        results: Evaluation results dictionary
        output_path: Path to save report
        sequence_name: Name of sequence
    """
    with open(output_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write(f"Visual Odometry Evaluation Report: {sequence_name}\n")
        f.write("="*80 + "\n\n")
        
        # ATE Section
        f.write("1. ABSOLUTE TRAJECTORY ERROR (ATE)\n")
        f.write("-" * 80 + "\n")
        ate = results['ate']
        f.write(f"   Root Mean Square Error (RMSE): {ate['rmse']:.4f} m\n")
        f.write(f"   Mean Error:                    {ate['mean']:.4f} m\n")
        f.write(f"   Median Error:                  {ate['median']:.4f} m\n")
        f.write(f"   Standard Deviation:            {ate['std']:.4f} m\n")
        f.write(f"   Minimum Error:                 {ate['min']:.4f} m\n")
        f.write(f"   Maximum Error:                 {ate['max']:.4f} m\n")
        f.write(f"   Scale Factor (alignment):      {ate['scale']:.6f}\n")
        f.write("\n")
        
        f.write("   Interpretation:\n")
        if ate['rmse'] < 0.5:
            f.write("   ✓ Excellent - Very accurate trajectory\n")
        elif ate['rmse'] < 1.0:
            f.write("   ✓ Good - Acceptable accuracy\n")
        elif ate['rmse'] < 2.0:
            f.write("   ⚠ Fair - Moderate drift\n")
        else:
            f.write("   ✗ Poor - Significant drift\n")
        f.write("\n\n")
        
        # RPE Section
        f.write("2. RELATIVE POSE ERROR (RPE)\n")
        f.write("-" * 80 + "\n")
        
        for step_name, rpe in results['rpe'].items():
            step = rpe['step']
            f.write(f"\n   Step Size: {step} frame(s)\n")
            f.write(f"   {'─' * 76}\n")
            
            f.write(f"   Translation Error:\n")
            f.write(f"     RMSE:   {rpe['translation']['rmse']:.6f} m\n")
            f.write(f"     Mean:   {rpe['translation']['mean']:.6f} m\n")
            f.write(f"     Median: {rpe['translation']['median']:.6f} m\n")
            f.write(f"     Std:    {rpe['translation']['std']:.6f} m\n")
            f.write(f"     Min:    {rpe['translation']['min']:.6f} m\n")
            f.write(f"     Max:    {rpe['translation']['max']:.6f} m\n")
            f.write(f"\n")
            
            f.write(f"   Rotation Error:\n")
            f.write(f"     RMSE:   {rpe['rotation']['rmse']:.6f} degrees\n")
            f.write(f"     Mean:   {rpe['rotation']['mean']:.6f} degrees\n")
            f.write(f"     Median: {rpe['rotation']['median']:.6f} degrees\n")
            f.write(f"     Std:    {rpe['rotation']['std']:.6f} degrees\n")
            f.write(f"     Min:    {rpe['rotation']['min']:.6f} degrees\n")
            f.write(f"     Max:    {rpe['rotation']['max']:.6f} degrees\n")
            f.write(f"\n")
            
            # Interpretation per step
            trans_rmse = rpe['translation']['rmse']
            rot_rmse = rpe['rotation']['rmse']
            
            f.write(f"   Interpretation (Step {step}):\n")
            if trans_rmse < 0.01 and rot_rmse < 0.5:
                f.write("   ✓ Excellent local consistency\n")
            elif trans_rmse < 0.05 and rot_rmse < 1.0:
                f.write("   ✓ Good local consistency\n")
            elif trans_rmse < 0.1 and rot_rmse < 2.0:
                f.write("   ⚠ Moderate local drift\n")
            else:
                f.write("   ✗ Significant local drift\n")
            f.write("\n")
        
        f.write("\n")
        
        # Trajectory Statistics
        f.write("3. TRAJECTORY STATISTICS\n")
        f.write("-" * 80 + "\n")
        f.write(f"   Number of Poses:       {results['num_poses']}\n")
        f.write(f"   Estimated Length:      {results['length']['estimated']:.2f} m\n")
        f.write(f"   Ground Truth Length:   {results['length']['ground_truth']:.2f} m\n")
        
        length_error = abs(results['length']['estimated'] - results['length']['ground_truth'])
        length_error_pct = (length_error / results['length']['ground_truth']) * 100
        f.write(f"   Length Error:          {length_error:.2f} m ({length_error_pct:.2f}%)\n")
        f.write("\n")
        
        f.write("   Scale Error Interpretation:\n")
        if length_error_pct < 2.0:
            f.write("   ✓ Excellent scale estimation\n")
        elif length_error_pct < 5.0:
            f.write("   ✓ Good scale estimation\n")
        elif length_error_pct < 10.0:
            f.write("   ⚠ Moderate scale drift\n")
        else:
            f.write("   ✗ Significant scale drift\n")
        f.write("\n\n")
        
        # Summary
        f.write("4. OVERALL SUMMARY\n")
        f.write("-" * 80 + "\n")
        
        # Calculate overall score
        ate_score = min(ate['rmse'] / 2.0, 1.0)  # Normalize to [0, 1]
        rpe_score = min(results['rpe']['step_1']['translation']['rmse'] / 0.1, 1.0)
        scale_score = min(length_error_pct / 10.0, 1.0)
        overall_score = 1.0 - (ate_score * 0.5 + rpe_score * 0.3 + scale_score * 0.2)
        
        f.write(f"   Overall Quality Score: {overall_score*100:.1f}/100\n\n")
        
        if overall_score > 0.9:
            f.write("   ★★★★★ Excellent performance\n")
        elif overall_score > 0.75:
            f.write("   ★★★★☆ Good performance\n")
        elif overall_score > 0.5:
            f.write("   ★★★☆☆ Acceptable performance\n")
        elif overall_score > 0.25:
            f.write("   ★★☆☆☆ Poor performance\n")
        else:
            f.write("   ★☆☆☆☆ Very poor performance\n")
        
        f.write("\n")
        f.write("="*80 + "\n")
        f.write("End of Report\n")
        f.write("="*80 + "\n")


if __name__ == "__main__":
    # Example usage
    print("Visual Odometry Evaluation Module")
    print("Import this module to evaluate camera trajectories.")
    print("\nExample:")
    print("  from evaluation.odometry_evaluation import evaluate_odometry, print_evaluation_results")
    print("  results = evaluate_odometry(estimated_poses, gt_poses)")
    print("  print_evaluation_results(results, 'Sequence 00')")
