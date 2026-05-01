"""
main.py — Entry point for the Datathon-VinUni Sales Forecasting Pipeline.

Usage
-----
    # Full pipeline with Optuna tuning (100 trials per model):
    python main.py

    # Quick run with default params (skip tuning):
    python main.py --skip-tuning

    # Custom settings:
    python main.py --n-trials 50 --n-folds 4
"""

import argparse
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.train import run_pipeline


def main():
    parser = argparse.ArgumentParser(
        description="Datathon-VinUni Sales Forecasting Pipeline (Stacking Ensemble)"
    )
    parser.add_argument(
        "--data-dir", type=str, default=None,
        help="Path to CSV data directory. Default: searches in ../DATA or ./data"
    )
    parser.add_argument(
        "--output-dir", type=str, default="models",
        help="Directory for saved models and submission (default: models/)"
    )
    parser.add_argument(
        "--log-dir", type=str, default="logs",
        help="Directory for log files (default: logs/)"
    )
    parser.add_argument(
        "--n-trials", type=int, default=100,
        help="Optuna trials per model (default: 100)"
    )
    parser.add_argument(
        "--n-folds", type=int, default=5,
        help="Number of CV folds (default: 5)"
    )
    parser.add_argument(
        "--val-months", type=int, default=6,
        help="Months per validation fold (default: 6)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--skip-tuning", action="store_true",
        help="Skip Optuna tuning, use default params"
    )

    args = parser.parse_args()

    # Resolve data directory
    data_dir = args.data_dir
    if data_dir is None:
        # Try common locations
        candidates = [
            os.path.join(os.path.dirname(__file__), "data"),
            os.path.join(os.path.dirname(__file__), "..", "DATA"),
        ]
        for c in candidates:
            if os.path.isdir(c) and os.path.exists(os.path.join(c, "sales.csv")):
                data_dir = c
                break

    if data_dir is None or not os.path.isdir(data_dir):
        print("ERROR: Data directory not found. Use --data-dir to specify.", file=sys.stderr)
        sys.exit(1)

    print(f"Data directory: {os.path.abspath(data_dir)}")
    print(f"Output directory: {os.path.abspath(args.output_dir)}")
    print(f"Settings: n_trials={args.n_trials}, n_folds={args.n_folds}, "
          f"val_months={args.val_months}, seed={args.seed}, "
          f"skip_tuning={args.skip_tuning}")
    print("=" * 70)

    submission = run_pipeline(
        data_dir=data_dir,
        output_dir=args.output_dir,
        log_dir=args.log_dir,
        n_trials=args.n_trials,
        n_folds=args.n_folds,
        val_months=args.val_months,
        seed=args.seed,
        skip_tuning=args.skip_tuning,
    )

    print("\nDone! Submission preview:")
    print(submission.head(10).to_string(index=False))
    print(f"\nSubmission file: {os.path.abspath(args.output_dir)}/submission.csv")


if __name__ == "__main__":
    main()
