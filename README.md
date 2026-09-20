# Stereo Vision and Visual Odometry on KITTI

Classical computer vision system implementing dense depth estimation and camera trajectory estimation  no deep learning.
## test 
## Project Structure

\proj/
 run_stereo_evaluation.py    # Part A: Stereo depth evaluation pipeline
 run_odometry_pipeline.py    # Part B: Visual odometry pipeline
 src/
    depth/
       stereo_matching.py  # Block matching (SAD/SSD/NCC/SGBM) + post-processing
    odometry/
       visual_odometry.py  # Feature-based VO + RANSAC + PnP
    utils/
        kitti_loader.py     # KITTI dataset loaders
        visualization.py   # Plotting and visualization utilities
 evaluation/
    depth_evaluation.py     # Depth metrics: EPE, D1, bad-pixel rate, MAE, RMSE
    odometry_evaluation.py  # VO metrics: ATE, RPE
 Odometry/                   # KITTI Odometry dataset (sequences 00-10)
 Stereo/                     # KITTI Stereo 2015 dataset
 results/                    # Auto-generated outputs
 config.py                   # Dataset paths
 requirements.txt            # Python dependencies
 QUICKSTART.md               # Quick start guide
 RUN_COMMANDS.md             # Full command reference
\
## Installation

\\ash
pip install -r requirements.txt
\
Requires Python >= 3.7 and OpenCV with contrib modules.

## Datasets

- **Stereo**: \./Stereo/\  KITTI Stereo 2015 (~200 rectified stereo pairs with GT disparity)
- **Odometry**: \./Odometry/\  KITTI Odometry sequences 00-10 with GT poses

## Part A: Stereo Depth Estimation

Compute disparity maps using block matching with SAD, SSD, NCC, or SGBM cost functions, then convert to metric depth via:

    Z = (f * B) / d

where f = focal length (pixels), B = stereo baseline (meters), d = disparity (pixels).

**Calibration** is read per-frame from \Stereo/calib_cam_to_cam/{idx:06d}.txt\:
- f = P_rect_02[0,0]
- B = |P_rect_03[0,3]| / f

\\ash
# Quick test (5 frames)
python run_stereo_evaluation.py --data_path ./Stereo --num_frames 5 --cost SAD

# Ablation: compare cost functions
python run_stereo_evaluation.py --data_path ./Stereo --num_frames 10 --cost SAD --output_dir ./results/stereo_SAD
python run_stereo_evaluation.py --data_path ./Stereo --num_frames 10 --cost SGBM --output_dir ./results/stereo_SGBM

# With post-processing (LR consistency + median filter + hole filling)
python run_stereo_evaluation.py --data_path ./Stereo --num_frames 10 --cost SGBM --postprocess
\
**Outputs** saved to esults/stereo_evaluation/\:
- \metrics.json\  aggregated bad-pixel rate, MAE, RMSE
- \per_frame_metrics.csv\  per-frame breakdown
- \detailed_summary.txt\  human-readable report
- \disparity_maps/\  predicted disparity .npy files
- \depth_maps/\  metric depth .npy files
- \error_maps/\  error visualizations
- \isualizations/\  5-panel comparison: left image, predicted disparity, GT disparity, error map, depth

## Part B: Visual Odometry

Feature-based VO with two motion estimation methods:

| Method | Description | Scale |
|--------|-------------|-------|
| essential | Essential matrix + RANSAC | Relative (up-to-scale) |
| pnp | PnP with stereo depth + RANSAC | Metric |

**Calibration** is read from \Odometry/{seq}/calib.txt\  projection matrix P0 provides focal length and baseline.

\\ash
# Quick test (50 frames, PnP method)
python run_odometry_pipeline.py --data_path ./Odometry --sequence 00 --end 50 --method pnp

# Ablation: RANSAC effect
python run_odometry_pipeline.py --data_path ./Odometry --sequence 00 --end 100 --method essential --no_ransac --output_dir ./results/vo_ablation/essential_no_ransac
python run_odometry_pipeline.py --data_path ./Odometry --sequence 00 --end 100 --method essential --output_dir ./results/vo_ablation/essential_ransac

# Ablation: stereo scale effect
python run_odometry_pipeline.py --data_path ./Odometry --sequence 00 --end 100 --method pnp --output_dir ./results/vo_ablation/pnp_stereo_scale
\
**Outputs** saved to esults/odometry/seq_{N}/\:
- \estimated_trajectory.npy\  estimated poses (Nx4x4)
- \	rajectory.png\  estimated vs GT trajectory
- \	rajectory_comparison.png\  multi-view comparison
- \evaluation_results.json\  ATE and RPE metrics
- \evaluation_report.txt\  full evaluation report
- \te_errors.png\, pe_errors_step_{1,5,10}.png\  error plots
- \matches/\  feature match visualizations with inliers highlighted

## Evaluation Metrics

**Depth:**
- Bad-pixel rate: fraction of pixels with |error| > 3px
- MAE: mean absolute disparity error
- RMSE: root mean squared error

**Visual Odometry:**
- ATE (Absolute Trajectory Error): global trajectory drift
- RPE (Relative Pose Error): local motion accuracy at steps 1, 5, 10

## Processing Times (approximate)

| Task | Frames | Time |
|------|--------|------|
| Stereo SAD, scale=0.5 | 10 | ~1s |
| Stereo SGBM, scale=0.5 | 10 | ~2s |
| Stereo NCC, scale=0.5 | 10 | ~3-5min |
| VO PnP + ORB | 100 | ~3min |
| VO PnP + SIFT | 100 | ~5min |

## Dependencies

`
numpy >= 1.21.0
opencv-python >= 4.5.0
opencv-contrib-python >= 4.5.0
matplotlib >= 3.4.0
scipy >= 1.7.0
scikit-image >= 0.18.0
tqdm >= 4.62.0
pandas >= 1.3.0
Pillow >= 8.3.0
