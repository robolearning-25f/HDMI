#!/usr/bin/env python3
import os
import re

import pandas as pd
import wandb


PROJECT = "DP-Next-HDMI/hdmi"
RUN_ID = "zrs2q1l5"
CSV_OUT = "env0-3_point_errors.csv"


def export_metrics(run):
    # Build metric keys
    metric_keys = []
    for env in range(4):  # env_0 ~ env_3
        for pt in [0, 1]:
            for dim in [2, 3]:
                metric_keys.append(f"env_{env}/point_{pt}_error_{dim}d")

    # Fetch history with only these columns
    hist = run.history(keys=metric_keys, pandas=True)
    # Save as CSV
    hist.to_csv(CSV_OUT, index=False)
    print(f"Saved metrics CSV -> {CSV_OUT}")


def download_tracker_viz_images(run):
    files = list(run.files())
    for env in range(4):
        env_dir = f"env_{env}/tracker_viz"
        os.makedirs(env_dir, exist_ok=True)

        pattern = re.compile(rf"env_{env}/tracker_viz")
        matched = [f for f in files if pattern.search(f.name)]

        if not matched:
            print(f"No files found for {env_dir}")
            continue

        for f in matched:
            f.download(root=env_dir, replace=True)
        print(f"Downloaded {len(matched)} files -> {env_dir}")


def main():
    api = wandb.Api()
    run = api.run(f"{PROJECT}/{RUN_ID}")

    export_metrics(run)
    download_tracker_viz_images(run)


if __name__ == "__main__":
    main()
