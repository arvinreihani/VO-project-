"""
Evaluation module initialization
"""

from .depth_evaluation import (
    evaluate_disparity,
    compute_bad_pixel_rate,
    compute_mae,
    compute_rmse,
    print_evaluation_results as print_depth_results
)

from .odometry_evaluation import (
    evaluate_odometry,
    compute_ate,
    compute_rpe,
    align_trajectories,
    print_evaluation_results as print_odometry_results
)

__all__ = [
    'evaluate_disparity',
    'compute_bad_pixel_rate',
    'compute_mae',
    'compute_rmse',
    'print_depth_results',
    'evaluate_odometry',
    'compute_ate',
    'compute_rpe',
    'align_trajectories',
    'print_odometry_results'
]
