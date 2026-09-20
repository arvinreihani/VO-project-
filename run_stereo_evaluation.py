"""
Stereo Depth Evaluation Pipeline

Evaluates stereo matching algorithms on the KITTI Stereo 2015 dataset.
Computes disparity with SAD/SSD/NCC/SGBM, converts to metric depth,
and evaluates against ground truth (bad-pixel rate, MAE, RMSE).

Usage:
    python run_stereo_evaluation.py --data_path ./Stereo --num_frames 10 --cost SAD
    python run_stereo_evaluation.py --data_path ./Stereo --num_frames 10 --cost SGBM --postprocess
"""

import os
import sys
import argparse
import numpy as np
import cv2
import time
import json
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.depth.stereo_matching import StereoMatcher
from src.utils.kitti_loader import KITTIStereo2015Loader, disparity_to_depth
from evaluation.depth_evaluation import (
    compute_bad_pixel_rate,
    compute_mae,
    compute_rmse,
    evaluate_disparity
)
from src.utils.visualization import visualize_disparity_comparison, visualize_depth


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Evaluate stereo matching on KITTI Stereo 2015 dataset',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Dataset parameters
    parser.add_argument('--data_path', type=str,
                        default='./Stereo',
                        help='Path to KITTI Stereo 2015 dataset')
    
    # Image selection
    parser.add_argument('--frames', type=str, default='0',
                        help='Frame indices to process (comma-separated or range, e.g., "0,5,10" or "0-10")')
    parser.add_argument('--num_frames', type=int, default=None,
                        help='Number of frames to process (overrides --frames if set)')
    
    # Stereo matching parameters
    parser.add_argument('--cost', type=str, default='SAD',
                        choices=['SAD', 'SSD', 'NCC', 'SGBM'],
                        help='Cost function for stereo matching (SGBM is recommended for best quality)')
    parser.add_argument('--window', type=int, default=7,
                        help='Block matching window size (odd number, default 7 for speed)')
    parser.add_argument('--max_disp', type=int, default=128,
                        help='Maximum disparity search range')
    parser.add_argument('--min_disp', type=int, default=0,
                        help='Minimum disparity')
    parser.add_argument('--scale', type=float, default=0.5,
                        help='Image scale factor for faster processing (0.5 = half size, 1.0 = full)')
    
    # Post-processing
    parser.add_argument('--no_postprocess', action='store_true',
                        help='Disable post-processing (default: post-processing disabled for speed)')
    parser.add_argument('--postprocess', action='store_true',
                        help='Enable post-processing (LR consistency, median filter, hole filling)')
    parser.add_argument('--lr_threshold', type=float, default=1.0,
                        help='Left-right consistency check threshold')
    parser.add_argument('--median_size', type=int, default=5,
                        help='Median filter kernel size')
    
    # Evaluation parameters
    parser.add_argument('--threshold', type=float, default=3.0,
                        help='Bad pixel threshold (in pixels) for evaluation')
    parser.add_argument('--use_noc', action='store_true',
                        help='Evaluate on non-occluded pixels only')
    
    # Comparison modes
    parser.add_argument('--ablation', action='store_true',
                        help='Run ablation study (compare all cost functions)')
    parser.add_argument('--compare_postprocess', action='store_true',
                        help='Compare with/without post-processing')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default='./results/stereo_evaluation',
                        help='Directory to save results')
    parser.add_argument('--save_disparity', action='store_true', default=True,
                        help='Save disparity maps as numpy arrays (enabled by default)')
    parser.add_argument('--no_save_disparity', action='store_true',
                        help='Disable saving disparity maps')
    parser.add_argument('--visualize', action='store_true', default=True,
                        help='Generate visualization plots (enabled by default)')
    parser.add_argument('--save_error_maps', action='store_true', default=True,
                        help='Save error maps for detailed analysis (enabled by default)')
    parser.add_argument('--save_all_details', action='store_true', default=True,
                        help='Save all intermediate results with full details (enabled by default)')
    
    return parser.parse_args()


def parse_frame_indices(frames_str, num_frames_arg, total_available):
    """
    Parse frame indices from string argument.
    
    Args:
        frames_str: String like "0,5,10" or "0-10"
        num_frames_arg: Number of frames to process (if set)
        total_available: Total number of frames available
        
    Returns:
        List of frame indices
    """
    if num_frames_arg is not None:
        return list(range(min(num_frames_arg, total_available)))
    
    if '-' in frames_str and ',' not in frames_str:
        # Range format: "0-10"
        start, end = map(int, frames_str.split('-'))
        return list(range(start, min(end + 1, total_available)))
    else:
        # Comma-separated: "0,5,10"
        indices = [int(x.strip()) for x in frames_str.split(',')]
        return [i for i in indices if i < total_available]


def load_ground_truth_disparity(loader, idx, use_noc=False):
    """
    Load ground truth disparity map.
    
    Args:
        loader: KITTIStereo2015Loader instance
        idx: Frame index
        use_noc: If True, use non-occluded disparity
        
    Returns:
        Ground truth disparity map (numpy array)
    """
    try:
        if use_noc:
            disp_path = os.path.join(loader.disp_noc_dir, f"{idx:06d}_10.png")
        else:
            disp_path = os.path.join(loader.disp_occ_dir, f"{idx:06d}_10.png")
        
        if not os.path.exists(disp_path):
            print(f"Warning: Ground truth disparity not found: {disp_path}")
            return None
        
        # Load disparity (KITTI format: divide by 256)
        disp_gt = cv2.imread(disp_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 256.0
        return disp_gt
    except Exception as e:
        print(f"Error loading ground truth disparity: {e}")
        return None


def evaluate_single_frame(matcher, loader, idx, args):
    """
    Evaluate stereo matching on a single frame.
    
    Args:
        matcher: StereoMatcher instance
        loader: KITTIStereo2015Loader instance
        idx: Frame index
        args: Command line arguments
        
    Returns:
        Dictionary with results
    """
    # Load images
    left_img, right_img = loader.get_image_pair(idx)
    
    # Load ground truth
    disp_gt = load_ground_truth_disparity(loader, idx, use_noc=args.use_noc)
    
    if disp_gt is None:
        print(f"Skipping frame {idx}: No ground truth available")
        return None
    
    # Convert to grayscale
    if len(left_img.shape) == 3:
        left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
    else:
        left_gray = left_img
        right_gray = right_img
    
    # Downsample if requested (for speed)
    original_shape = left_gray.shape
    if args.scale != 1.0:
        print(f"  Downsampling images by factor {args.scale} ({original_shape[1]}x{original_shape[0]} -> "
              f"{int(original_shape[1]*args.scale)}x{int(original_shape[0]*args.scale)})")
        left_gray = cv2.resize(left_gray, None, fx=args.scale, fy=args.scale, interpolation=cv2.INTER_LINEAR)
        right_gray = cv2.resize(right_gray, None, fx=args.scale, fy=args.scale, interpolation=cv2.INTER_LINEAR)
        disp_gt_scaled = cv2.resize(disp_gt, None, fx=args.scale, fy=args.scale, interpolation=cv2.INTER_NEAREST)
        disp_gt_scaled = disp_gt_scaled * args.scale  # Scale disparity values
    
    # Compute disparity
    print(f"  Computing disparity for frame {idx} ({left_gray.shape[1]}x{left_gray.shape[0]}, window={args.window}, max_disp={args.max_disp})...")
    start_time = time.time()
    disp_pred = matcher.compute_disparity(left_gray, right_gray, verbose=True)
    
    # Upscale disparity back to original resolution if downsampled
    if args.scale != 1.0:
        print(f"  Upscaling disparity back to original resolution...")
        disp_pred = cv2.resize(disp_pred, (original_shape[1], original_shape[0]), interpolation=cv2.INTER_LINEAR)
        disp_pred = disp_pred / args.scale  # Unscale disparity values
        # Use original ground truth for evaluation
    else:
        disp_gt_scaled = disp_gt
    
    # Apply post-processing if enabled
    if args.postprocess:
        print(f"  Applying post-processing (LR consistency + median filter + hole filling)...")
        disp_pred = matcher.post_process(
            left_gray if args.scale == 1.0 else cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY),
            right_gray if args.scale == 1.0 else cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY),
            disp_pred,
            consistency_threshold=args.lr_threshold,
            median_kernel=args.median_size,
            verbose=False
        )
    
    elapsed_time = time.time() - start_time
    print(f"  ✓ Frame {idx} processed in {elapsed_time:.1f}s")
    
    # Create valid mask (where ground truth is available)
    valid_mask = disp_gt > 0
    
    # Compute metrics
    eval_results = evaluate_disparity(
        disp_pred, disp_gt,
        thresholds=[args.threshold]
    )
    
    # Extract metrics in flat format
    # Note: evaluate_disparity returns bad_pixel_rate as percentage (0-100),
    # but rest of code expects ratio (0-1), so divide by 100
    metrics = {
        'bad_pixel_rate': eval_results['bad_pixel'][f'{args.threshold}px'] / 100.0,
        'mae': eval_results['mae'],
        'rmse': eval_results['rmse'],
        'processing_time': elapsed_time,
        'frame_idx': idx
    }
    
    return {
        'metrics': metrics,
        'disp_pred': disp_pred,
        'disp_gt': disp_gt,
        'left_img': left_img,
        'right_img': right_img,
        'valid_mask': valid_mask
    }


def run_single_configuration(args, frame_indices, loader):
    """
    Run evaluation with a single configuration.
    
    Args:
        args: Command line arguments
        frame_indices: List of frame indices to process
        loader: KITTIStereo2015Loader instance
        
    Returns:
        Dictionary with aggregated results
    """
    print(f"\n{'='*60}")
    print(f"Configuration: {args.cost}, window={args.window}, max_disp={args.max_disp}")
    print(f"Image scale: {args.scale}x ({int(1242*args.scale)}x{int(375*args.scale)} typical)")
    print(f"Post-processing: {args.postprocess} (off by default for speed)")
    est_time = len(frame_indices) * (0.5 if args.scale == 0.5 and not args.postprocess else (2 if args.postprocess else 1)) / 60
    print(f"Estimated time: ~{est_time:.1f} minutes ({len(frame_indices)} frames)")
    print(f"{'='*60}\n")
    
    # Create stereo matcher
    # Note: OpenCV's StereoBM only supports SAD, so for SSD/NCC we use Python implementation
    use_opencv = args.cost.upper() in ['SAD', 'SGBM']
    matcher = StereoMatcher(
        cost_function=args.cost.upper(),
        window_size=args.window,
        max_disparity=args.max_disp,
        use_opencv=use_opencv
    )
    
    # Process each frame
    all_results = []
    all_metrics = []
    
    for idx in tqdm(frame_indices, desc=f"Processing frames ({args.cost})"):
        result = evaluate_single_frame(matcher, loader, idx, args)
        if result is not None:
            all_results.append(result)
            all_metrics.append(result['metrics'])
    
    if not all_metrics:
        print("No valid frames processed!")
        return None
    
    # Aggregate metrics
    aggregated = {
        'bad_pixel_rate': np.mean([m['bad_pixel_rate'] for m in all_metrics]),
        'mae': np.mean([m['mae'] for m in all_metrics]),
        'rmse': np.mean([m['rmse'] for m in all_metrics]),
        'processing_time_mean': np.mean([m['processing_time'] for m in all_metrics]),
        'processing_time_std': np.std([m['processing_time'] for m in all_metrics]),
        'num_frames': len(all_metrics),
        'per_frame_metrics': all_metrics
    }
    
    return {
        'aggregated': aggregated,
        'frame_results': all_results,
        'config': {
            'cost_function': args.cost,
            'window_size': args.window,
            'max_disparity': args.max_disp,
            'post_processing': not args.no_postprocess,
            'threshold': args.threshold
        }
    }


def save_results(results, args, loader, config_name=""):
    """
    Save evaluation results to disk with all details.
    
    Args:
        results: Results dictionary
        args: Command line arguments
        loader: KITTIStereo2015Loader instance (for calibration)
        config_name: Optional name suffix for config
    """
    # Create output directory
    output_dir = Path(args.output_dir) / config_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save aggregated metrics as JSON
    metrics_file = output_dir / 'metrics.json'
    with open(metrics_file, 'w') as f:
        # Convert numpy types to Python types for JSON
        metrics_json = {
            'aggregated': {
                k: float(v) if isinstance(v, (np.floating, np.integer)) else v
                for k, v in results['aggregated'].items()
                if k != 'per_frame_metrics'
            },
            'config': results['config']
        }
        json.dump(metrics_json, f, indent=2)
    
    print(f"\n✓ Metrics saved to: {metrics_file}")
    
    # Save per-frame metrics as CSV
    csv_file = output_dir / 'per_frame_metrics.csv'
    with open(csv_file, 'w') as f:
        f.write("frame_idx,bad_pixel_rate,mae,rmse,processing_time\n")
        for m in results['aggregated']['per_frame_metrics']:
            f.write(f"{m['frame_idx']},{m['bad_pixel_rate']:.4f},"
                   f"{m['mae']:.4f},{m['rmse']:.4f},{m['processing_time']:.3f}\n")
    
    print(f"✓ Per-frame metrics saved to: {csv_file}")
    
    # Save disparity maps (enabled by default)
    save_disp = args.save_disparity and not args.no_save_disparity if hasattr(args, 'no_save_disparity') else args.save_disparity
    if save_disp:
        disp_dir = output_dir / 'disparity_maps'
        disp_dir.mkdir(exist_ok=True)
        
        for result in results['frame_results']:
            idx = result['metrics']['frame_idx']
            np.save(disp_dir / f'disp_pred_{idx:06d}.npy', result['disp_pred'])
            np.save(disp_dir / f'disp_gt_{idx:06d}.npy', result['disp_gt'])
        
        print(f"✓ Disparity maps saved to: {disp_dir} ({len(results['frame_results'])} frames)")

    # Save depth maps (converted from disparity using stereo geometry)
    if save_disp:
        depth_dir = output_dir / 'depth_maps'
        depth_dir.mkdir(exist_ok=True)

        for result in results['frame_results']:
            idx = result['metrics']['frame_idx']

            # Get calibration for this frame
            try:
                calib_raw = loader.load_calibration(idx)
                p2_vals = list(map(float, calib_raw['P_rect_02'].split()))
                fx = p2_vals[0]
                p3_vals = list(map(float, calib_raw['P_rect_03'].split()))
                baseline = abs(p3_vals[3] / fx)
            except Exception:
                fx = 721.5377
                baseline = 0.5327

            depth_pred = disparity_to_depth(result['disp_pred'], fx, baseline)

            # Save raw depth as numpy only (visualization is inside comparison panel)
            np.save(depth_dir / f'depth_pred_{idx:06d}.npy', depth_pred)

        print(f"✓ Depth maps saved to: {depth_dir} ({len(results['frame_results'])} frames)")

    # Save error maps (NEW - enabled by default)
    if getattr(args, 'save_error_maps', True):
        error_dir = output_dir / 'error_maps'
        error_dir.mkdir(exist_ok=True)
        
        for result in results['frame_results']:
            idx = result['metrics']['frame_idx']
            
            # Compute error map
            disp_pred = result['disp_pred']
            disp_gt = result['disp_gt']
            valid_mask = disp_gt > 0
            
            error_map = np.zeros_like(disp_pred)
            error_map[valid_mask] = np.abs(disp_pred[valid_mask] - disp_gt[valid_mask])
            
            # Save as numpy and visualization
            np.save(error_dir / f'error_{idx:06d}.npy', error_map)
            
            # Create error visualization
            fig, ax = plt.subplots(figsize=(12, 6))
            im = ax.imshow(error_map, cmap='hot', vmin=0, vmax=10)
            ax.set_title(f'Disparity Error Map - Frame {idx}')
            ax.axis('off')
            plt.colorbar(im, ax=ax, label='Error (pixels)')
            fig.savefig(error_dir / f'error_vis_{idx:06d}.png', dpi=150, bbox_inches='tight')
            plt.close(fig)
        
        print(f"✓ Error maps saved to: {error_dir} ({len(results['frame_results'])} frames)")
    
    # Generate visualizations (enabled by default)
    if args.visualize:
        vis_dir = output_dir / 'visualizations'
        vis_dir.mkdir(exist_ok=True)
        
        for result in results['frame_results']:
            idx = result['metrics']['frame_idx']

            # Get calibration for depth in visualization
            fx_vis, baseline_vis = None, None
            try:
                calib_raw = loader.load_calibration(idx)
                p2_vals = list(map(float, calib_raw['P_rect_02'].split()))
                fx_vis = p2_vals[0]
                p3_vals = list(map(float, calib_raw['P_rect_03'].split()))
                baseline_vis = abs(p3_vals[3] / fx_vis)
            except Exception:
                fx_vis, baseline_vis = 721.5377, 0.5327

            # Create comparison visualization (with depth map)
            fig = visualize_disparity_comparison(
                result['disp_pred'],
                result['disp_gt'],
                result['left_img'],
                title=f"Frame {idx} - {args.cost}",
                focal_length=fx_vis,
                baseline=baseline_vis
            )

            fig.savefig(vis_dir / f'comparison_{idx:06d}.png', dpi=150, bbox_inches='tight')
            plt.close(fig)
        
        print(f"✓ Visualizations saved to: {vis_dir} ({len(results['frame_results'])} frames)")
    
    # Generate comprehensive evaluation report (replaces simple summary)
    if getattr(args, 'save_all_details', True):
        from evaluation.depth_evaluation import generate_depth_evaluation_report
        
        summary_file = output_dir / 'detailed_summary.txt'
        generate_depth_evaluation_report(results, str(summary_file), config_name)
        print(f"✓ Comprehensive evaluation report saved to: {summary_file}")
    
    # Generate detailed analysis plots
    print("\nGenerating detailed analysis plots...")
    
    from evaluation.depth_evaluation import (
        plot_error_distribution, plot_per_frame_metrics, 
        plot_disparity_statistics
    )
    
    # 1. Per-frame metrics plot
    if len(results['aggregated']['per_frame_metrics']) > 1:
        metrics_plot = output_dir / 'per_frame_metrics.png'
        plot_per_frame_metrics(results['aggregated']['per_frame_metrics'], 
                              save_path=str(metrics_plot))
        print(f"  ✓ Per-frame metrics plot: {metrics_plot}")
    
    # 2-4. Individual frame analysis (for each frame)
    analysis_dir = output_dir / 'detailed_analysis'
    analysis_dir.mkdir(exist_ok=True)
    
    for i, result in enumerate(results['frame_results'][:3]):  # First 3 frames for detail
        idx = result['metrics']['frame_idx']
        
        # Error distribution
        error_dist_path = analysis_dir / f'error_distribution_{idx:06d}.png'
        plot_error_distribution(
            result['disp_pred'], result['disp_gt'],
            save_path=str(error_dist_path),
            title=f"Error Distribution - Frame {idx}"
        )
        
        # Disparity statistics
        disp_stats_path = analysis_dir / f'disparity_statistics_{idx:06d}.png'
        plot_disparity_statistics(
            result['disp_pred'], result['disp_gt'],
            save_path=str(disp_stats_path)
        )
    
    print(f"  ✓ Detailed analysis plots saved to: {analysis_dir}")
    print(f"    (Generated for first {min(3, len(results['frame_results']))} frames)")


def run_ablation_study(args, frame_indices, loader):
    """
    Run ablation study comparing different cost functions.
    
    Args:
        args: Command line arguments
        frame_indices: List of frame indices to process
        loader: KITTIStereo2015Loader instance
    """
    print("\n" + "="*60)
    print("ABLATION STUDY: Comparing Cost Functions")
    print("="*60)
    
    cost_functions = ['SAD', 'SSD', 'NCC']
    all_results = {}
    
    for cost in cost_functions:
        # Create modified args for this configuration
        config_args = argparse.Namespace(**vars(args))
        config_args.cost = cost
        
        # Run evaluation
        results = run_single_configuration(config_args, frame_indices, loader)
        if results is not None:
            all_results[cost] = results
            
            # Save individual results
            save_results(results, args, loader, config_name=f"ablation_{cost}")
    
    if not all_results:
        print("No valid results from ablation study!")
        return
    
    # Create comparison summary
    print("\n" + "="*60)
    print("ABLATION STUDY RESULTS")
    print("="*60)
    print(f"{'Method':<10} {'Bad Pixel %':<15} {'MAE':<10} {'RMSE':<10} {'Time (s)':<10}")
    print("-"*60)
    
    for cost, results in all_results.items():
        agg = results['aggregated']
        print(f"{cost:<10} {agg['bad_pixel_rate']*100:>14.2f}% "
              f"{agg['mae']:>9.2f} {agg['rmse']:>9.2f} "
              f"{agg['processing_time_mean']:>9.2f}")
    
    # Create comparison plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    methods = list(all_results.keys())
    bad_pixel = [all_results[m]['aggregated']['bad_pixel_rate'] * 100 for m in methods]
    mae = [all_results[m]['aggregated']['mae'] for m in methods]
    rmse = [all_results[m]['aggregated']['rmse'] for m in methods]
    
    axes[0].bar(methods, bad_pixel, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    axes[0].set_ylabel('Bad Pixel Rate (%)')
    axes[0].set_title('Lower is Better')
    axes[0].grid(axis='y', alpha=0.3)
    
    axes[1].bar(methods, mae, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    axes[1].set_ylabel('MAE (pixels)')
    axes[1].set_title('Lower is Better')
    axes[1].grid(axis='y', alpha=0.3)
    
    axes[2].bar(methods, rmse, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    axes[2].set_ylabel('RMSE (pixels)')
    axes[2].set_title('Lower is Better')
    axes[2].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    comp_dir = Path(args.output_dir) / 'ablation_comparison'
    comp_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(comp_dir / 'cost_function_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nComparison plot saved to: {comp_dir / 'cost_function_comparison.png'}")


def run_postprocess_comparison(args, frame_indices, loader):
    """
    Compare results with and without post-processing.
    
    Args:
        args: Command line arguments
        frame_indices: List of frame indices to process
        loader: KITTIStereo2015Loader instance
    """
    print("\n" + "="*60)
    print("POST-PROCESSING COMPARISON")
    print("="*60)
    
    results_dict = {}
    
    for enable_pp in [False, True]:
        config_args = argparse.Namespace(**vars(args))
        config_args.no_postprocess = not enable_pp
        
        name = "with_postprocess" if enable_pp else "without_postprocess"
        
        results = run_single_configuration(config_args, frame_indices, loader)
        if results is not None:
            results_dict[name] = results
            save_results(results, args, loader, config_name=f"postprocess_{name}")
    
    if len(results_dict) != 2:
        print("Could not complete post-processing comparison!")
        return
    
    # Print comparison
    print("\n" + "="*60)
    print("POST-PROCESSING COMPARISON RESULTS")
    print("="*60)
    print(f"{'Configuration':<25} {'Bad Pixel %':<15} {'MAE':<10} {'RMSE':<10}")
    print("-"*65)
    
    for name, results in results_dict.items():
        agg = results['aggregated']
        display_name = "With Post-Processing" if "with" in name else "Without Post-Processing"
        print(f"{display_name:<25} {agg['bad_pixel_rate']*100:>14.2f}% "
              f"{agg['mae']:>9.2f} {agg['rmse']:>9.2f}")
    
    # Calculate improvement
    no_pp = results_dict['without_postprocess']['aggregated']
    with_pp = results_dict['with_postprocess']['aggregated']
    
    bp_improve = (no_pp['bad_pixel_rate'] - with_pp['bad_pixel_rate']) / no_pp['bad_pixel_rate'] * 100
    mae_improve = (no_pp['mae'] - with_pp['mae']) / no_pp['mae'] * 100
    rmse_improve = (no_pp['rmse'] - with_pp['rmse']) / no_pp['rmse'] * 100
    
    print("\nImprovement from post-processing:")
    print(f"  Bad Pixel Rate: {bp_improve:+.1f}%")
    print(f"  MAE: {mae_improve:+.1f}%")
    print(f"  RMSE: {rmse_improve:+.1f}%")


def main():
    """Main function."""
    args = parse_args()
    
    print("\n" + "="*60)
    print("KITTI Stereo 2015 Evaluation")
    print("="*60)
    print(f"Dataset path: {args.data_path}")
    print(f"Cost function: {args.cost}")
    print(f"Window size: {args.window}")
    print(f"Max disparity: {args.max_disp}")
    print(f"Post-processing: {not args.no_postprocess}")
    print(f"Evaluation threshold: {args.threshold} pixels")
    print(f"Output directory: {args.output_dir}")
    print("="*60 + "\n")
    
    # Load dataset
    print("Loading KITTI Stereo 2015 dataset...")
    try:
        loader = KITTIStereo2015Loader(args.data_path)
        total_frames = loader.num_samples
        print(f"Found {total_frames} stereo pairs\n")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print(f"Make sure {args.data_path} contains the KITTI Stereo 2015 dataset")
        return 1
    
    # Parse frame indices
    frame_indices = parse_frame_indices(args.frames, args.num_frames, total_frames)
    print(f"Processing {len(frame_indices)} frames: {frame_indices[:10]}" + 
          ("..." if len(frame_indices) > 10 else "") + "\n")
    
    # Run evaluation based on mode
    if args.ablation:
        run_ablation_study(args, frame_indices, loader)
    elif args.compare_postprocess:
        run_postprocess_comparison(args, frame_indices, loader)
    else:
        # Standard single configuration evaluation
        results = run_single_configuration(args, frame_indices, loader)
        
        if results is not None:
            # Print summary
            agg = results['aggregated']
            print("\n" + "="*60)
            print("EVALUATION RESULTS")
            print("="*60)
            print(f"Bad Pixel Rate: {agg['bad_pixel_rate']*100:.2f}%")
            print(f"MAE: {agg['mae']:.2f} pixels")
            print(f"RMSE: {agg['rmse']:.2f} pixels")
            print(f"Processing time: {agg['processing_time_mean']:.2f} ± {agg['processing_time_std']:.2f} seconds/frame")
            print(f"Total frames: {agg['num_frames']}")
            print("="*60 + "\n")
            
            # Save results
            save_results(results, args, loader)
    
    print("\n✅ Evaluation complete!")
    return 0


if __name__ == '__main__':
    sys.exit(main())
