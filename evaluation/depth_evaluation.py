"""
Depth Evaluation Script
Evaluates disparity/depth estimation on KITTI Stereo 2015 dataset.
Computes metrics: Bad-pixel rate, MAE, RMSE
"""

import numpy as np
import sys
import os
from typing import Dict, Tuple

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def compute_bad_pixel_rate(predicted: np.ndarray,
                           ground_truth: np.ndarray,
                           threshold: float = 3.0) -> Dict[str, float]:
    """
    Compute bad-pixel rate.
    
    A pixel is considered "bad" if the absolute error exceeds threshold pixels.
    
    Args:
        predicted: Predicted disparity map
        ground_truth: Ground truth disparity map
        threshold: Error threshold in pixels
    
    Returns:
        Dictionary with bad-pixel statistics
    """
    # Valid pixels are those with ground truth > 0
    valid_mask = ground_truth > 0
    
    if np.sum(valid_mask) == 0:
        return {'bad_pixel_rate': 0.0, 'num_valid': 0, 'num_bad': 0}
    
    # Compute absolute error
    error = np.abs(predicted - ground_truth)
    
    # Count bad pixels
    bad_pixels = (error > threshold) & valid_mask
    num_bad = np.sum(bad_pixels)
    num_valid = np.sum(valid_mask)
    
    bad_pixel_rate = (num_bad / num_valid) * 100.0
    
    return {
        'bad_pixel_rate': bad_pixel_rate,
        'num_valid': num_valid,
        'num_bad': num_bad
    }


def compute_mae(predicted: np.ndarray, ground_truth: np.ndarray) -> float:
    """
    Compute Mean Absolute Error.
    
    Args:
        predicted: Predicted disparity map
        ground_truth: Ground truth disparity map
    
    Returns:
        MAE value
    """
    valid_mask = ground_truth > 0
    
    if np.sum(valid_mask) == 0:
        return 0.0
    
    error = np.abs(predicted[valid_mask] - ground_truth[valid_mask])
    mae = np.mean(error)
    
    return mae


def compute_rmse(predicted: np.ndarray, ground_truth: np.ndarray) -> float:
    """
    Compute Root Mean Squared Error.
    
    Args:
        predicted: Predicted disparity map
        ground_truth: Ground truth disparity map
    
    Returns:
        RMSE value
    """
    valid_mask = ground_truth > 0
    
    if np.sum(valid_mask) == 0:
        return 0.0
    
    error = predicted[valid_mask] - ground_truth[valid_mask]
    rmse = np.sqrt(np.mean(error ** 2))
    
    return rmse


def compute_percentage_error(predicted: np.ndarray,
                            ground_truth: np.ndarray,
                            threshold_percent: float = 5.0) -> float:
    """
    Compute percentage of pixels with error > threshold_percent of ground truth.
    
    Args:
        predicted: Predicted disparity map
        ground_truth: Ground truth disparity map
        threshold_percent: Error threshold as percentage
    
    Returns:
        Percentage of bad pixels
    """
    valid_mask = ground_truth > 0
    
    if np.sum(valid_mask) == 0:
        return 0.0
    
    # Compute relative error
    error = np.abs(predicted[valid_mask] - ground_truth[valid_mask])
    relative_error = (error / ground_truth[valid_mask]) * 100.0
    
    # Count pixels with error > threshold
    bad_pixels = relative_error > threshold_percent
    percentage = (np.sum(bad_pixels) / len(bad_pixels)) * 100.0
    
    return percentage


def evaluate_disparity(predicted: np.ndarray,
                      ground_truth: np.ndarray,
                      thresholds: list = [1.0, 2.0, 3.0, 5.0]) -> Dict:
    """
    Comprehensive disparity evaluation.
    
    Args:
        predicted: Predicted disparity map
        ground_truth: Ground truth disparity map
        thresholds: List of thresholds for bad-pixel rate
    
    Returns:
        Dictionary with all metrics
    """
    results = {}
    
    # Bad-pixel rates at different thresholds
    results['bad_pixel'] = {}
    for thresh in thresholds:
        bp_stats = compute_bad_pixel_rate(predicted, ground_truth, thresh)
        results['bad_pixel'][f'{thresh}px'] = bp_stats['bad_pixel_rate']
    
    # MAE and RMSE
    results['mae'] = compute_mae(predicted, ground_truth)
    results['rmse'] = compute_rmse(predicted, ground_truth)
    
    # Percentage errors
    results['percentage_error'] = {}
    for percent in [5.0, 10.0, 15.0]:
        pe = compute_percentage_error(predicted, ground_truth, percent)
        results['percentage_error'][f'{percent}%'] = pe
    
    # Number of valid pixels
    valid_mask = ground_truth > 0
    results['num_valid_pixels'] = int(np.sum(valid_mask))
    results['total_pixels'] = ground_truth.size
    
    return results


def print_evaluation_results(results: Dict, method_name: str = "Method"):
    """
    Print evaluation results in a formatted way.
    
    Args:
        results: Dictionary with evaluation results
        method_name: Name of the method being evaluated
    """
    print(f"\n{'='*60}")
    print(f"Evaluation Results: {method_name}")
    print(f"{'='*60}")
    
    print(f"\nBad-Pixel Rate:")
    for thresh, rate in results['bad_pixel'].items():
        print(f"  {thresh:>5}: {rate:6.2f}%")
    
    print(f"\nError Metrics:")
    print(f"  MAE:  {results['mae']:.3f} pixels")
    print(f"  RMSE: {results['rmse']:.3f} pixels")
    
    print(f"\nPercentage Error:")
    for percent, rate in results['percentage_error'].items():
        print(f"  >{percent:>4}: {rate:6.2f}%")
    
    print(f"\nPixel Statistics:")
    print(f"  Valid pixels: {results['num_valid_pixels']:,} / {results['total_pixels']:,}")
    print(f"  Coverage: {(results['num_valid_pixels']/results['total_pixels']*100):.2f}%")
    
    print(f"{'='*60}\n")


def plot_error_distribution(predicted: np.ndarray, ground_truth: np.ndarray, 
                           save_path: str = None, title: str = "Error Distribution"):
    """
    Plot error distribution histogram.
    
    Args:
        predicted: Predicted disparity map
        ground_truth: Ground truth disparity map
        save_path: Path to save plot
        title: Plot title
    """
    import matplotlib.pyplot as plt
    
    valid_mask = ground_truth > 0
    errors = predicted[valid_mask] - ground_truth[valid_mask]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram of errors
    ax = axes[0]
    ax.hist(errors, bins=100, color='steelblue', alpha=0.7, edgecolor='black')
    ax.axvline(np.mean(errors), color='r', linestyle='--', linewidth=2, 
               label=f'Mean: {np.mean(errors):.2f}px')
    ax.axvline(np.median(errors), color='orange', linestyle='--', linewidth=2,
               label=f'Median: {np.median(errors):.2f}px')
    ax.set_xlabel('Error (pixels)', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Error Distribution', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Cumulative distribution
    ax = axes[1]
    abs_errors = np.abs(errors)
    sorted_errors = np.sort(abs_errors)
    cumulative = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors) * 100
    ax.plot(sorted_errors, cumulative, linewidth=2, color='steelblue')
    
    # Mark thresholds
    for thresh in [1, 2, 3, 5]:
        pct = np.sum(abs_errors <= thresh) / len(abs_errors) * 100
        ax.axvline(thresh, color='red', linestyle='--', alpha=0.5)
        ax.text(thresh, 50, f'{thresh}px\n{pct:.1f}%', ha='center', fontsize=9)
    
    ax.set_xlabel('Absolute Error (pixels)', fontsize=12)
    ax.set_ylabel('Cumulative Percentage (%)', fontsize=12)
    ax.set_title('Cumulative Error Distribution', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 10)
    
    plt.suptitle(title, fontsize=15, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_per_frame_metrics(per_frame_metrics: list, save_path: str = None):
    """
    Plot per-frame metrics over frames.
    
    Args:
        per_frame_metrics: List of per-frame metric dictionaries
        save_path: Path to save plot
    """
    import matplotlib.pyplot as plt
    
    frames = [m['frame_idx'] for m in per_frame_metrics]
    bad_pixel = [m['bad_pixel_rate'] * 100 for m in per_frame_metrics]
    mae = [m['mae'] for m in per_frame_metrics]
    rmse = [m['rmse'] for m in per_frame_metrics]
    times = [m['processing_time'] for m in per_frame_metrics]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Bad pixel rate
    ax = axes[0, 0]
    ax.plot(frames, bad_pixel, 'o-', linewidth=2, markersize=6, color='steelblue')
    ax.axhline(np.mean(bad_pixel), color='r', linestyle='--', linewidth=2,
               label=f'Mean: {np.mean(bad_pixel):.2f}%')
    ax.set_xlabel('Frame Index', fontsize=11)
    ax.set_ylabel('Bad Pixel Rate (%)', fontsize=11)
    ax.set_title('Bad Pixel Rate per Frame', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # MAE
    ax = axes[0, 1]
    ax.plot(frames, mae, 'o-', linewidth=2, markersize=6, color='darkorange')
    ax.axhline(np.mean(mae), color='r', linestyle='--', linewidth=2,
               label=f'Mean: {np.mean(mae):.2f}px')
    ax.set_xlabel('Frame Index', fontsize=11)
    ax.set_ylabel('MAE (pixels)', fontsize=11)
    ax.set_title('Mean Absolute Error per Frame', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # RMSE
    ax = axes[1, 0]
    ax.plot(frames, rmse, 'o-', linewidth=2, markersize=6, color='green')
    ax.axhline(np.mean(rmse), color='r', linestyle='--', linewidth=2,
               label=f'Mean: {np.mean(rmse):.2f}px')
    ax.set_xlabel('Frame Index', fontsize=11)
    ax.set_ylabel('RMSE (pixels)', fontsize=11)
    ax.set_title('Root Mean Square Error per Frame', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Processing time
    ax = axes[1, 1]
    ax.plot(frames, times, 'o-', linewidth=2, markersize=6, color='purple')
    ax.axhline(np.mean(times), color='r', linestyle='--', linewidth=2,
               label=f'Mean: {np.mean(times):.2f}s')
    ax.set_xlabel('Frame Index', fontsize=11)
    ax.set_ylabel('Processing Time (seconds)', fontsize=11)
    ax.set_title('Processing Time per Frame', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.suptitle('Per-Frame Performance Metrics', fontsize=15, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_disparity_statistics(predicted: np.ndarray, ground_truth: np.ndarray,
                             save_path: str = None):
    """
    Plot disparity value statistics and comparisons.
    
    Args:
        predicted: Predicted disparity map
        ground_truth: Ground truth disparity map
        save_path: Path to save plot
    """
    import matplotlib.pyplot as plt
    
    valid_mask = ground_truth > 0
    pred_valid = predicted[valid_mask]
    gt_valid = ground_truth[valid_mask]
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # Scatter plot: predicted vs ground truth
    ax = axes[0]
    # Sample for visualization (too many points)
    sample_size = min(10000, len(pred_valid))
    indices = np.random.choice(len(pred_valid), sample_size, replace=False)
    ax.scatter(gt_valid[indices], pred_valid[indices], alpha=0.3, s=1, c='steelblue')
    
    # Perfect prediction line
    max_val = max(gt_valid.max(), pred_valid.max())
    ax.plot([0, max_val], [0, max_val], 'r--', linewidth=2, label='Perfect prediction')
    
    ax.set_xlabel('Ground Truth Disparity (pixels)', fontsize=11)
    ax.set_ylabel('Predicted Disparity (pixels)', fontsize=11)
    ax.set_title('Predicted vs Ground Truth', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    
    # Histogram comparison
    ax = axes[1]
    ax.hist(gt_valid, bins=50, alpha=0.5, color='green', label='Ground Truth', density=True)
    ax.hist(pred_valid, bins=50, alpha=0.5, color='blue', label='Predicted', density=True)
    ax.set_xlabel('Disparity (pixels)', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('Disparity Distribution Comparison', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Error vs disparity
    ax = axes[2]
    errors = np.abs(pred_valid - gt_valid)
    
    # Bin by disparity ranges
    disp_bins = np.linspace(0, gt_valid.max(), 20)
    bin_indices = np.digitize(gt_valid, disp_bins)
    
    mean_errors = []
    bin_centers = []
    for i in range(1, len(disp_bins)):
        mask = bin_indices == i
        if np.sum(mask) > 10:
            mean_errors.append(np.mean(errors[mask]))
            bin_centers.append((disp_bins[i-1] + disp_bins[i]) / 2)
    
    ax.plot(bin_centers, mean_errors, 'o-', linewidth=2, markersize=6, color='steelblue')
    ax.set_xlabel('Ground Truth Disparity (pixels)', fontsize=11)
    ax.set_ylabel('Mean Absolute Error (pixels)', fontsize=11)
    ax.set_title('Error vs Disparity Range', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def generate_depth_evaluation_report(results_dict: dict, output_path: str, 
                                    config_name: str = "Configuration"):
    """
    Generate comprehensive text report of depth evaluation.
    
    Args:
        results_dict: Dictionary with all evaluation results
        output_path: Path to save report
        config_name: Name of configuration
    """
    agg = results_dict['aggregated']
    config = results_dict['config']
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("="*80 + "\n")
        f.write(f"STEREO DEPTH ESTIMATION EVALUATION REPORT\n")
        f.write(f"{config_name}\n")
        f.write("="*80 + "\n\n")
        
        # Configuration
        f.write("1. CONFIGURATION\n")
        f.write("-" * 80 + "\n")
        f.write(f"   Cost Function:      {config['cost_function']}\n")
        f.write(f"   Window Size:        {config['window_size']}\n")
        f.write(f"   Max Disparity:      {config['max_disparity']}\n")
        f.write(f"   Post-processing:    {config['post_processing']}\n")
        f.write(f"   Eval Threshold:     {config['threshold']} pixels\n")
        f.write("\n\n")
        
        # Summary metrics
        f.write("2. SUMMARY METRICS\n")
        f.write("-" * 80 + "\n")
        f.write(f"   Bad Pixel Rate:     {agg['bad_pixel_rate']*100:.2f}%\n")
        f.write(f"   Mean Absolute Error (MAE):   {agg['mae']:.3f} pixels\n")
        f.write(f"   Root Mean Square Error (RMSE): {agg['rmse']:.3f} pixels\n")
        f.write(f"   Processing Time:    {agg['processing_time_mean']:.2f} ± {agg['processing_time_std']:.2f} sec/frame\n")
        f.write(f"   Number of Frames:   {agg['num_frames']}\n")
        f.write("\n")
        
        # Quality assessment
        f.write("   Quality Assessment:\n")
        if agg['bad_pixel_rate'] < 0.05:
            f.write("   ★★★★★ Excellent - State-of-the-art quality\n")
        elif agg['bad_pixel_rate'] < 0.15:
            f.write("   ★★★★☆ Good - High quality results\n")
        elif agg['bad_pixel_rate'] < 0.30:
            f.write("   ★★★☆☆ Acceptable - Moderate quality\n")
        elif agg['bad_pixel_rate'] < 0.50:
            f.write("   ★★☆☆☆ Poor - Significant errors\n")
        else:
            f.write("   ★☆☆☆☆ Very Poor - Major issues\n")
        f.write("\n\n")
        
        # Performance analysis
        f.write("3. PERFORMANCE ANALYSIS\n")
        f.write("-" * 80 + "\n")
        
        fps = 1.0 / agg['processing_time_mean']
        f.write(f"   Throughput:         {fps:.2f} FPS\n")
        
        if fps > 10:
            f.write("   ✓ Real-time capable (>10 FPS)\n")
        elif fps > 5:
            f.write("   ⚡ Near real-time (5-10 FPS)\n")
        else:
            f.write("   ⚠ Offline processing required (<5 FPS)\n")
        
        f.write("\n")
        
        # Speedup estimate
        if config['cost_function'] == 'SGBM':
            f.write("   Note: Using optimized OpenCV SGBM\n")
            f.write("   • ~1800x faster than pure Python implementation\n")
            f.write("   • Utilizes SIMD instructions and multi-threading\n")
        f.write("\n\n")
        
        # Per-frame details
        f.write("4. PER-FRAME DETAILS\n")
        f.write("-" * 80 + "\n")
        f.write(f"{'Frame':<10} {'Bad Pixel %':<15} {'MAE (px)':<12} {'RMSE (px)':<12} {'Time (s)':<10}\n")
        f.write("-" * 80 + "\n")
        
        for m in agg['per_frame_metrics']:
            f.write(f"{m['frame_idx']:<10d} {m['bad_pixel_rate']*100:>14.2f}% "
                   f"{m['mae']:>11.3f} {m['rmse']:>11.3f} {m['processing_time']:>9.2f}\n")
        
        f.write("\n\n")
        
        # Statistics
        f.write("5. STATISTICAL SUMMARY\n")
        f.write("-" * 80 + "\n")
        
        bad_pixels = [m['bad_pixel_rate'] for m in agg['per_frame_metrics']]
        maes = [m['mae'] for m in agg['per_frame_metrics']]
        rmses = [m['rmse'] for m in agg['per_frame_metrics']]
        
        f.write(f"   Bad Pixel Rate:\n")
        f.write(f"     Min:    {np.min(bad_pixels)*100:.2f}%\n")
        f.write(f"     Max:    {np.max(bad_pixels)*100:.2f}%\n")
        f.write(f"     Mean:   {np.mean(bad_pixels)*100:.2f}%\n")
        f.write(f"     Median: {np.median(bad_pixels)*100:.2f}%\n")
        f.write(f"     Std:    {np.std(bad_pixels)*100:.2f}%\n")
        f.write("\n")
        
        f.write(f"   MAE (pixels):\n")
        f.write(f"     Min:    {np.min(maes):.3f}\n")
        f.write(f"     Max:    {np.max(maes):.3f}\n")
        f.write(f"     Mean:   {np.mean(maes):.3f}\n")
        f.write(f"     Median: {np.median(maes):.3f}\n")
        f.write(f"     Std:    {np.std(maes):.3f}\n")
        f.write("\n")
        
        f.write(f"   RMSE (pixels):\n")
        f.write(f"     Min:    {np.min(rmses):.3f}\n")
        f.write(f"     Max:    {np.max(rmses):.3f}\n")
        f.write(f"     Mean:   {np.mean(rmses):.3f}\n")
        f.write(f"     Median: {np.median(rmses):.3f}\n")
        f.write(f"     Std:    {np.std(rmses):.3f}\n")
        
        f.write("\n")
        f.write("="*80 + "\n")
        f.write("End of Report\n")
        f.write("="*80 + "\n")


if __name__ == "__main__":
    # Example usage
    print("Depth Evaluation Module")
    print("Import this module to evaluate disparity maps.")
    print("\nExample:")
    print("  from evaluation.depth_evaluation import evaluate_disparity, print_evaluation_results")
    print("  results = evaluate_disparity(predicted_disp, gt_disp)")
    print("  print_evaluation_results(results, 'My Method')")
