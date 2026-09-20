# Quick Start

## 1. Install dependencies

```bash
pip install -r requirements.txt
```

## 2. Part A  Stereo Depth (5 frames, ~5 seconds)

```bash
python run_stereo_evaluation.py --data_path ./Stereo --num_frames 5 --cost SAD
```

Results saved to `results/stereo_evaluation/`

## 3. Part B  Visual Odometry (50 frames, ~1 minute)

```bash
python run_odometry_pipeline.py --data_path ./Odometry --sequence 00 --end 50 --method pnp
```

Results saved to `results/odometry/seq_00/`

## Ablation Studies

```bash
# Depth: compare cost functions
python run_stereo_evaluation.py --data_path ./Stereo --num_frames 10 --cost SAD  --output_dir ./results/stereo_SAD
python run_stereo_evaluation.py --data_path ./Stereo --num_frames 10 --cost SGBM --output_dir ./results/stereo_SGBM

# VO: RANSAC effect
python run_odometry_pipeline.py --sequence 00 --end 100 --method essential --no_ransac --output_dir ./results/vo_ablation/no_ransac
python run_odometry_pipeline.py --sequence 00 --end 100 --method essential             --output_dir ./results/vo_ablation/with_ransac

# VO: stereo scale effect
python run_odometry_pipeline.py --sequence 00 --end 100 --method pnp --output_dir ./results/vo_ablation/pnp_scale
```

See `RUN_COMMANDS.md` for all parameters and options.