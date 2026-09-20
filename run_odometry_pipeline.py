"""
Visual Odometry Pipeline

Estimates camera trajectory from KITTI Odometry stereo sequences.

Usage:
    python run_odometry_pipeline.py --data_path ./Odometry --sequence 00 --method pnp
    python run_odometry_pipeline.py --data_path ./Odometry --sequence 00 --method essential
    python run_odometry_pipeline.py --data_path ./Odometry --sequence 00 --start 0 --end 100
"""

import argparse
import os
import sys
import numpy as np
import cv2
from tqdm import tqdm
import time

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.utils.kitti_loader import KITTIOdometryLoader
from src.odometry.visual_odometry import VisualOdometry
from src.depth.stereo_matching import StereoMatcher
from src.utils.visualization import visualize_trajectory, visualize_matches
from evaluation.odometry_evaluation import evaluate_odometry, print_evaluation_results


def run_visual_odometry(loader, vo, start_frame, end_frame, method,
                       compute_disparity, output_dir, visualize_freq=10,
                       use_ransac=True):
    """
    Run visual odometry on a sequence.
    
    Args:
        loader: KITTI data loader
        vo: Visual odometry object
        start_frame: Starting frame index
        end_frame: Ending frame index
        method: 'pnp' or 'essential'
        compute_disparity: Whether to compute disparity for depth
        output_dir: Output directory for visualizations
        visualize_freq: Frequency of match visualization
        use_ransac: Whether to use RANSAC for outlier rejection
    """
    print(f"\n{'='*70}")
    print(f"Running Visual Odometry")
    print(f"Method: {method.upper()}")
    print(f"RANSAC: {'Enabled' if use_ransac else 'Disabled'}")
    print(f"Frames: {start_frame} to {end_frame}")
    print(f"{'='*70}\n")
    
    # Create stereo matcher if needed
    if method == 'pnp' and compute_disparity:
        print("Initializing stereo matcher for depth estimation...")
        matcher = StereoMatcher(
            window_size=11,
            max_disparity=128,
            cost_function='NCC'
        )
    
    # Reset VO
    vo.reset()
    
    # Load first frame
    prev_left, prev_right = loader.get_stereo_pair(start_frame, color=False)
    
    # Compute disparity for first frame if needed
    prev_disparity = None
    if method == 'pnp' and compute_disparity:
        print(f"Computing disparity for frame {start_frame}...")
        prev_disparity = matcher.compute_disparity(prev_left, prev_right, verbose=False)
    
    # Process frames
    matches_dir = os.path.join(output_dir, "matches")
    os.makedirs(matches_dir, exist_ok=True)
    
    start_time = time.time()
    
    for frame_idx in tqdm(range(start_frame + 1, end_frame + 1), desc="Processing frames"):
        # Load current frame
        curr_left, curr_right = loader.get_stereo_pair(frame_idx, color=False)
        
        # Compute disparity for current frame if using PnP
        curr_disparity = None
        if method == 'pnp' and compute_disparity:
            curr_disparity = matcher.compute_disparity(curr_left, curr_right, verbose=False)
        
        # Estimate motion
        result = vo.process_frame(
            prev_left, curr_left,
            disparity_map_prev=prev_disparity,
            method=method,
            use_ransac=use_ransac
        )
        
        if result['success']:
            # Update pose
            vo.update_pose(result['R'], result['t'])
            
            # Save matches (by default saves all frames, can be changed with visualize_freq)
            if frame_idx % visualize_freq == 0:
                match_path = os.path.join(matches_dir, f"matches_{frame_idx:06d}.png")
                visualize_matches(
                    prev_left, curr_left,
                    result['pts1'], result['pts2'],
                    result['inliers'],
                    title=f"Frame {frame_idx} - {result['num_inliers']}/{result['total_matches']} inliers",
                    save_path=match_path,
                    max_matches=100  # Show more matches for detailed analysis
                )
            
            # Save detailed metrics per frame
            metrics_data = [
                f"{frame_idx},{result['num_inliers']},{result['total_matches']},"
                f"{result['num_inliers']/result['total_matches']*100:.2f}\n"
            ]
        else:
            print(f"Warning: Failed to estimate motion for frame {frame_idx}")
            # Use identity transformation
            vo.update_pose(np.eye(3), np.zeros((3, 1)))
        
        # Update previous frame
        prev_left = curr_left
        prev_disparity = curr_disparity
    
    elapsed_time = time.time() - start_time
    num_frames = end_frame - start_frame
    fps = num_frames / elapsed_time
    
    print(f"\nProcessing complete!")
    print(f"Time: {elapsed_time:.2f} seconds ({fps:.2f} FPS)")
    
    # Get trajectory
    estimated_trajectory = vo.get_trajectory()
    
    return estimated_trajectory


def main():
    parser = argparse.ArgumentParser(description='Run visual odometry pipeline')
    parser.add_argument('--data_path', type=str, required=True,
                       help='Path to KITTI odometry dataset')
    parser.add_argument('--sequence', type=str, default='00',
                       help='Sequence number (00-10 for evaluation)')
    parser.add_argument('--start', type=int, default=0,
                       help='Start frame index')
    parser.add_argument('--end', type=int, default=-1,
                       help='End frame index (-1 for all frames)')
    parser.add_argument('--method', type=str, default='pnp',
                       choices=['pnp', 'essential'],
                       help='Motion estimation method')
    parser.add_argument('--feature', type=str, default='ORB',
                       choices=['ORB', 'SIFT', 'AKAZE'],
                       help='Feature detector type')
    parser.add_argument('--max_features', type=int, default=3000,
                       help='Maximum number of features to detect')
    parser.add_argument('--output_dir', type=str, default='./results/odometry',
                       help='Output directory')
    parser.add_argument('--no_disparity', action='store_true',
                       help='Do not compute disparity (use pre-computed or skip)')
    parser.add_argument('--no_ransac', action='store_true',
                       help='Disable RANSAC outlier rejection (to show effect of RANSAC)')
    parser.add_argument('--visualize_freq', type=int, default=1,
                       help='Frequency of match visualization (default: 1 = save all)')
    parser.add_argument('--evaluate', action='store_true',
                       help='Evaluate against ground truth')
    parser.add_argument('--save_all_matches', action='store_true', default=True,
                       help='Save feature matches for all frames (enabled by default)')
    parser.add_argument('--save_detailed_metrics', action='store_true', default=True,
                       help='Save detailed per-frame metrics (enabled by default)')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = os.path.join(args.output_dir, f"seq_{args.sequence}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Load dataset
    print(f"Loading KITTI Odometry sequence {args.sequence}...")
    loader = KITTIOdometryLoader(args.data_path, args.sequence)
    print(f"Loaded {loader.num_frames} frames")
    
    # Determine frame range
    start_frame = args.start
    end_frame = args.end if args.end != -1 else loader.num_frames - 1
    end_frame = min(end_frame, loader.num_frames - 1)
    
    # Get camera parameters
    params = loader.get_camera_params('P0')
    K = loader.get_intrinsics('P0')
    
    print(f"\nCamera Parameters:")
    print(f"  Focal length: {params['fx']:.2f} pixels")
    print(f"  Baseline: {params['baseline']:.4f} meters")
    print(f"  Principal point: ({params['cx']:.2f}, {params['cy']:.2f})")
    
    # Initialize Visual Odometry
    print(f"\nInitializing Visual Odometry...")
    print(f"  Feature detector: {args.feature}")
    print(f"  Max features: {args.max_features}")
    print(f"  Method: {args.method.upper()}")
    
    vo = VisualOdometry(
        K=K,
        baseline=params['baseline'],
        focal_length=params['fx'],
        feature_detector=args.feature,
        max_features=args.max_features
    )
    
    # Run visual odometry
    estimated_trajectory = run_visual_odometry(
        loader, vo,
        start_frame, end_frame,
        args.method,
        compute_disparity=(not args.no_disparity),
        output_dir=output_dir,
        visualize_freq=args.visualize_freq,
        use_ransac=(not args.no_ransac)
    )
    
    # Save trajectory
    traj_path = os.path.join(output_dir, "estimated_trajectory.npy")
    np.save(traj_path, estimated_trajectory)
    print(f"\nSaved trajectory to {traj_path}")
    
    # Check if ground truth is available
    gt_trajectory = None
    if loader.poses is not None:
        # Convert ground truth to 4x4
        gt_trajectory = np.zeros((end_frame - start_frame + 1, 4, 4))
        for i in range(len(gt_trajectory)):
            gt_trajectory[i] = loader.get_pose(start_frame + i)
    
    # Visualize trajectory (basic plot)
    traj_vis_path = os.path.join(output_dir, "trajectory.png")
    visualize_trajectory(
        estimated_trajectory,
        gt_trajectory,
        title=f"Trajectory - Sequence {args.sequence}",
        save_path=traj_vis_path
    )
    print(f"\nSaved trajectory visualization to {traj_vis_path}")
    
    # Always evaluate if ground truth is available (not just with --evaluate flag)
    if gt_trajectory is not None:
        print(f"\n{'='*70}")
        print("Evaluating against ground truth...")
        print(f"{'='*70}")
        
        results = evaluate_odometry(
            estimated_trajectory,
            gt_trajectory,
            rpe_steps=[1, 5, 10]
        )
        
        print_evaluation_results(results, f"Sequence {args.sequence}")
        
        # Save evaluation results
        import json
        results_path = os.path.join(output_dir, "evaluation_results.json")
        
        # Convert numpy arrays to lists for JSON serialization
        results_json = {}
        for key, value in results.items():
            if key == 'ate':
                results_json[key] = {k: float(v) if not isinstance(v, np.ndarray) else v.tolist() 
                                    for k, v in value.items()}
            elif key == 'rpe':
                results_json[key] = {}
                for step_key, step_value in value.items():
                    results_json[key][step_key] = {
                        'translation': {k: float(v) if not isinstance(v, np.ndarray) else v.tolist()
                                       for k, v in step_value['translation'].items()},
                        'rotation': {k: float(v) if not isinstance(v, np.ndarray) else v.tolist()
                                    for k, v in step_value['rotation'].items()},
                        'step': int(step_value['step'])
                    }
            else:
                results_json[key] = value
        
        with open(results_path, 'w') as f:
            json.dump(results_json, f, indent=2)
        
        print(f"\nSaved evaluation results to {results_path}")
        
        # Generate detailed plots and reports
        from evaluation.odometry_evaluation import (
            plot_ate_errors, plot_rpe_errors, plot_trajectory_comparison,
            generate_evaluation_report
        )
        
        print("\nGenerating detailed evaluation plots...")
        
        # 1. ATE error plot over time
        ate_plot_path = os.path.join(output_dir, "ate_errors.png")
        plot_ate_errors(results['ate'], save_path=ate_plot_path)
        print(f"  - ATE error plot: {ate_plot_path}")
        
        # 2. RPE error plots for each step
        for step_name, rpe_data in results['rpe'].items():
            rpe_plot_path = os.path.join(output_dir, f"rpe_errors_{step_name}.png")
            plot_rpe_errors(rpe_data, save_path=rpe_plot_path)
            print(f"  - RPE error plot ({step_name}): {rpe_plot_path}")
        
        # 3. Detailed trajectory comparison (3D views)
        traj_compare_path = os.path.join(output_dir, "trajectory_comparison.png")
        plot_trajectory_comparison(
            estimated_trajectory, gt_trajectory,
            results['ate'], save_path=traj_compare_path
        )
        print(f"  - Trajectory comparison: {traj_compare_path}")
        
        # 4. Generate comprehensive text report
        report_path = os.path.join(output_dir, "evaluation_report.txt")
        generate_evaluation_report(results, report_path, f"Sequence {args.sequence}")
        print(f"  - Detailed report: {report_path}")
    else:
        print("\nGround truth not available - skipping evaluation")
    
    print(f"\n{'='*70}")
    print("Visual odometry pipeline completed!")
    print(f"Results saved to: {output_dir}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()
