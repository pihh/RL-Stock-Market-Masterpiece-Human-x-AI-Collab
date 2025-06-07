import os
import tempfile
import pandas as pd
from src.experiment.experiment_tracker import ExperimentTracker, deep_hash

def test_save_and_load_runs():
    # Setup
    with tempfile.TemporaryDirectory() as tmpdir:
        test_csv = os.path.join(tmpdir, "test_experiments.csv")
        tracker = ExperimentTracker(project="test_proj")
        tracker.csv_path = test_csv  # Use temp csv
        
        config = {"param": 42, "reward": "sharpe"}
        results = {"test": {"sharpe": 1.2}, "train": {"sharpe": 1.5}}
        target_date = "2025-06-09"
        run_settings = {"mode": "in-sample"}
        
        tracker.save_run(config, results, target_date, run_settings)
        df = tracker.load_runs()
        assert not df.empty
        assert df.iloc[0]["hash"] == deep_hash(config)
        assert pd.notna(df.iloc[0]["results_json"])

def test_duplicate_runs_not_saved():
    # Setup
    with tempfile.TemporaryDirectory() as tmpdir:
        test_csv = os.path.join(tmpdir, "test_experiments.csv")
        tracker = ExperimentTracker(project="test_proj")
        tracker.csv_path = test_csv
        
        config = {"param": 42}
        results = {"test": {"sharpe": 1.2}}
        target_date = "2025-06-09"
        run_settings = {"mode": "in-sample"}
        
        tracker.save_run(config, results, target_date, run_settings)
        tracker.save_run(config, results, target_date, run_settings)  # Should skip as duplicate
        df = tracker.load_runs()
        assert len(df) == 1
