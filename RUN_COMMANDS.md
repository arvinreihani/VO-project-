# Run Commands Reference

## Datasets

```
Odometry/
  00/  image_0/  image_1/  calib.txt  00.txt
  ...
  10/

Stereo/
  image_2/           Left color images
  image_3/           Right color images
  disp_noc_0/        Ground truth disparity (non-occluded)
  calib_cam_to_cam/  Per-frame calibration files
```

---

## 1. Stereo Depth Evaluation

Script: `run_stereo_evaluation.py`

### Parameters

```
--data_path PATH           Path to Stereo dataset (default: ./Stereo)
--frames "0,5,10"          Specific frame indices (comma-separated)
--num_frames N             Evaluate first N frames
--cost {SAD|SSD|NCC|SGBM}  Matching cost function (default: SAD)
--window N                 Block matching window size, must be odd (default: 7)
--max_disp N               Maximum disparity search range (default: 128)
--scale FLOAT              Image scale factor (default: 0.5)
--postprocess              Enable LR consistency + median filter + hole filling
--threshold FLOAT          Bad-pixel threshold in pixels (default: 3.0)
--output_dir PATH          Output directory (default: ./results/stereo_evaluation)
--visualize                Save comparison visualizations (enabled by default)
```

### Examples

```bash
# Quick test - 5 frames, SAD
python run_stereo_evaluation.py --data_path ./Stereo --num_frames 5 --cost SAD

# Full evaluation - SGBM with post-processing
python run_stereo_evaluation.py --data_path ./Stereo --num_frames 10 --cost SGBM --postprocess

# Ablation: cost function comparison
python run_stereo_evaluation.py --data_path ./Stereo --num_frames 10 --cost SAD  --output_dir ./results/stereo_evaluation_SAD_7
python run_stereo_evaluation.py --data_path ./Stereo --num_frames 10 --cost SGBM --output_dir ./results/stereo_evaluation_SGBM_7

# Ablation: window size comparison
python run_stereo_evaluation.py --data_path ./Stereo --num_frames 10 --cost SGBM --window 7  --output_dir ./results/stereo_evaluation_SGBM_7
python run_stereo_evaluation.py --data_path ./Stereo --num_frames 10 --cost SGBM --window 11 --output_dir ./results/stereo_evaluation_SGBM_11
python run_stereo_evaluation.py --data_path ./Stereo --num_frames 10 --cost SGBM --window 21 --output_dir ./results/stereo_evaluation_SGBM_21
```

### Outputs

```
results/stereo_evaluation/
  metrics.json              Aggregated bad-pixel rate, MAE, RMSE
  per_frame_metrics.csv     Per-frame metrics
  detailed_summary.txt      Text report
  disparity_maps/           Predicted disparity (.npy)
  depth_maps/               Metric depth (.npy)
  error_maps/               Error visualizations (.png)
  visualizations/           5-panel comparison images
```

### Speed Reference

| Cost | Implementation | Speed (scale=0.5) |
|------|----------------|-------------------|
| SAD  | OpenCV         | ~0.04s/frame      |
| SGBM | OpenCV         | ~0.08s/frame      |
| SSD  | Python         | ~5-10s/frame      |
| NCC  | Python         | ~15-30s/frame     |

For SSD/NCC, use `--num_frames 3 --scale 0.1` to keep runtime manageable.

---

## 2. Visual Odometry

Script: `run_odometry_pipeline.py`

### Parameters

```
--data_path PATH            Path to Odometry dataset (required)
--sequence SEQ              Sequence number 00-10 (default: 00)
--start N                   Start frame index (default: 0)
--end N                     End frame index, -1 = all (default: -1)
--method {pnp|essential}    Motion estimation method (default: pnp)
--feature {ORB|SIFT|AKAZE}  Feature detector (default: ORB)
--max_features N            Maximum features to detect (default: 3000)
--no_disparity              Skip disparity computation (use with essential)
--no_ransac                 Disable RANSAC outlier rejection
--visualize_freq N          Save match images every N frames (default: 1)
--output_dir PATH           Output directory (default: ./results/odometry)
```

### Examples

```bash
# Quick test - 50 frames, PnP
python run_odometry_pipeline.py --data_path ./Odometry --sequence 00 --end 50 --method pnp

# Full sequence 00 with SIFT
python run_odometry_pipeline.py --data_path ./Odometry --sequence 00 --method pnp --feature SIFT

# Ablation: RANSAC effect
python run_odometry_pipeline.py --sequence 00 --end 100 --method essential --no_ransac --output_dir ./results/vo_ablation/essential_no_ransac
python run_odometry_pipeline.py --sequence 00 --end 100 --method essential            --output_dir ./results/vo_ablation/essential_ransac

# Ablation: stereo scale effect
python run_odometry_pipeline.py --sequence 00 --end 100 --method pnp --output_dir ./results/vo_ablation/pnp_stereo_scale

# Multiple sequences
python run_odometry_pipeline.py --data_path ./Odometry --sequence 01 --end 100 --method pnp
python run_odometry_pipeline.py --data_path ./Odometry --sequence 02 --end 100 --method pnp
```

### Outputs

```
results/odometry/seq_00/
  estimated_trajectory.npy    Estimated poses (Nx4x4)
  trajectory.png              Estimated vs GT trajectory plot
  trajectory_comparison.png   Multi-view comparison
  evaluation_results.json     ATE and RPE metrics
  evaluation_report.txt       Full text report
  ate_errors.png              ATE error over time
  rpe_errors_step_1.png       RPE at step 1
  rpe_errors_step_5.png       RPE at step 5
  rpe_errors_step_10.png      RPE at step 10
  matches/                    Feature match images (inliers highlighted)
```

### KITTI Odometry Sequences

| Seq | Frames | Environment      |
|-----|--------|------------------|
| 00  | 4541   | Urban (loop)     |
| 01  | 1101   | Highway          |
| 02  | 4661   | Urban + Highway  |
| 03  | 801    | Country road     |
| 04  | 271    | Country road     |
| 05  | 2761   | Urban            |
| 06  | 1101   | Urban            |
| 07  | 1101   | Urban            |
| 08  | 4071   | Urban + Highway  |
| 09  | 1591   | Urban            |
| 10  | 1201   | Urban            |

---

## Quick Reference

```bash
# Fastest stereo test (~5 seconds)
python run_stereo_evaluation.py --num_frames 5 --cost SAD --scale 0.5

# Fastest VO test (~1 minute)
python run_odometry_pipeline.py --sequence 00 --end 50 --method essential --feature ORB --no_disparity

# Best quality stereo
python run_stereo_evaluation.py --num_frames 20 --cost SGBM --window 11 --postprocess

# Best quality VO
python run_odometry_pipeline.py --sequence 00 --method pnp --feature SIFT --max_features 5000
```