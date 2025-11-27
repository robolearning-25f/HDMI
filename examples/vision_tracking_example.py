#!/usr/bin/env python
"""
Example script demonstrating vision-based object tracking in HDMI.

This shows how to:
1. Set up vision tracking in your environment
2. Access tracked points and object positions
3. Compare vision estimates with ground truth
4. Debug and visualize tracking
"""

import torch
import numpy as np
from pathlib import Path


def example_basic_vision_tracking():
    """Basic example of using vision tracking with HDMI."""
    from omegaconf import OmegaConf
    import hydra
    from active_adaptation.envs import make_env

    # Load config with vision tracking enabled
    cfg = OmegaConf.load("cfg/task/G1/hdmi/move_suitcase_vision.yaml")

    # Create environment
    env = make_env(cfg)

    # Get vision tracker reference
    vision_obs = env.observation_manager.observations["vision"]["cotracker_object_position"]
    print(f"Vision tracker initialized: {vision_obs.tracker}")

    # Reset environment
    obs = env.reset()

    # Run a few steps
    for step in range(100):
        # Random actions for demo
        action = torch.randn(env.num_envs, env.action_space.shape[0], device=env.device)

        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)

        # Access vision-based object position
        if "vision" in obs and "cotracker_object_position" in obs["vision"]:
            vision_pos = obs["vision"]["cotracker_object_position"]
            print(f"Step {step}: Vision object position: {vision_pos[0]}")

        # Compare with ground truth
        if hasattr(env.command_manager, 'object'):
            gt_pos_w = env.command_manager.object.data.root_link_pos_w
            vision_pos_w = env.command_manager.vision_object_pos_w

            error = torch.norm(vision_pos_w - gt_pos_w, dim=-1)
            print(f"Step {step}: Tracking error: {error.mean().item():.4f}m")

        if step % 20 == 0:
            print(f"\n--- Step {step} ---")
            print_tracking_stats(vision_obs)

    env.close()


def example_debug_tracking():
    """Example showing how to debug and visualize tracking."""
    from omegaconf import OmegaConf
    from active_adaptation.envs import make_env
    import matplotlib.pyplot as plt
    from PIL import Image

    cfg = OmegaConf.load("cfg/task/G1/hdmi/move_suitcase_vision.yaml")
    cfg.num_envs = 1  # Single env for debugging

    env = make_env(cfg)
    vision_obs = env.observation_manager.observations["vision"]["cotracker_object_position"]

    obs = env.reset()

    # Create output directory
    output_dir = Path("debug_tracking")
    output_dir.mkdir(exist_ok=True)

    for step in range(50):
        action = torch.randn(1, env.action_space.shape[0], device=env.device)
        obs, reward, terminated, truncated, info = env.step(action)

        # Get camera image and tracked points
        rgb_img = env.scene["tiled_camera"].data.output["rgb"][0].cpu().numpy()
        tracked_points = vision_obs.tracker.get_tracked_points()[0].cpu().numpy()
        visibility = vision_obs.tracker.get_visibility()[0].cpu().numpy()

        # Visualize tracking
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))
        ax.imshow(rgb_img)

        # Plot tracked points
        visible_points = tracked_points[visibility]
        invisible_points = tracked_points[~visibility]

        ax.scatter(visible_points[:, 0], visible_points[:, 1],
                  c='green', s=50, marker='o', label='Visible')
        ax.scatter(invisible_points[:, 0], invisible_points[:, 1],
                  c='red', s=50, marker='x', label='Lost')

        # Add mean position
        if len(visible_points) > 0:
            mean_pos = visible_points.mean(axis=0)
            ax.scatter(mean_pos[0], mean_pos[1],
                      c='blue', s=200, marker='+', linewidths=3, label='Mean')

        ax.legend()
        ax.set_title(f"Step {step}: {visibility.sum()}/{len(visibility)} points visible")

        plt.savefig(output_dir / f"tracking_{step:04d}.png")
        plt.close()

        if step % 10 == 0:
            print(f"Step {step}: Saved visualization")

    print(f"\nVisualizations saved to {output_dir}/")
    env.close()


def example_compare_tracking_methods():
    """Compare CoTracker vs simple tracking vs ground truth."""
    from omegaconf import OmegaConf
    from active_adaptation.envs import make_env

    results = {}

    for method in ["cotracker", "simple", "ground_truth"]:
        print(f"\n{'='*60}")
        print(f"Testing: {method}")
        print(f"{'='*60}")

        cfg = OmegaConf.load("cfg/task/G1/hdmi/move_suitcase_vision.yaml")
        cfg.num_envs = 16

        if method == "ground_truth":
            cfg.command.use_vision_tracking = False
        else:
            cfg.command.use_vision_tracking = True
            cfg.observation.vision.cotracker_object_position.use_cotracker = (method == "cotracker")

        env = make_env(cfg)
        obs = env.reset()

        errors = []
        for step in range(100):
            action = torch.randn(env.num_envs, env.action_space.shape[0], device=env.device)
            obs, reward, terminated, truncated, info = env.step(action)

            if method != "ground_truth":
                gt_pos = env.command_manager.object.data.root_link_pos_w
                vision_pos = env.command_manager.vision_object_pos_w
                error = torch.norm(vision_pos - gt_pos, dim=-1)
                errors.append(error.cpu().numpy())

        env.close()

        if errors:
            errors = np.array(errors)
            results[method] = {
                'mean': errors.mean(),
                'std': errors.std(),
                'max': errors.max()
            }

    # Print comparison
    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    for method, stats in results.items():
        print(f"\n{method.upper()}:")
        print(f"  Mean error: {stats['mean']:.4f}m")
        print(f"  Std error:  {stats['std']:.4f}m")
        print(f"  Max error:  {stats['max']:.4f}m")


def print_tracking_stats(vision_obs):
    """Print tracking statistics."""
    visibility = vision_obs.tracker.get_visibility()
    num_visible = visibility.sum(dim=1)

    print(f"Tracking stats across {len(num_visible)} environments:")
    print(f"  Mean visible points: {num_visible.float().mean():.1f}")
    print(f"  Min visible points:  {num_visible.min()}")
    print(f"  Max visible points:  {num_visible.max()}")
    print(f"  Tracking confidence: {vision_obs.tracking_confidence.mean():.3f}")


def example_curriculum_training():
    """Example of curriculum learning: gradually transition from GT to vision."""
    from omegaconf import OmegaConf
    from active_adaptation.envs import make_env

    cfg = OmegaConf.load("cfg/task/G1/hdmi/move_suitcase_vision.yaml")
    cfg.command.blend_vision_ground_truth = True

    # Curriculum schedule
    curriculum = [
        (0, 100, 0.0),      # Episodes 0-100: pure ground truth
        (100, 200, 0.25),   # Episodes 100-200: 25% vision
        (200, 300, 0.5),    # Episodes 200-300: 50% vision
        (300, 400, 0.75),   # Episodes 300-400: 75% vision
        (400, 1000, 1.0),   # Episodes 400+: pure vision
    ]

    env = make_env(cfg)

    episode = 0
    while episode < 1000:
        # Update vision weight based on curriculum
        for start, end, weight in curriculum:
            if start <= episode < end:
                env.command_manager.vision_weight = weight
                break

        print(f"Episode {episode}: vision_weight = {env.command_manager.vision_weight:.2f}")

        obs = env.reset()

        done = False
        step = 0
        while not done and step < 500:
            # Your policy here
            action = torch.randn(env.num_envs, env.action_space.shape[0], device=env.device)
            obs, reward, terminated, truncated, info = env.step(action)

            done = terminated.any() or truncated.any()
            step += 1

        episode += 1

    env.close()


def example_multi_object_tracking():
    """Example for future: tracking multiple objects (placeholder)."""
    print("Multi-object tracking not yet implemented.")
    print("To implement:")
    print("1. Modify vision_tracking.py to track multiple objects")
    print("2. Use separate trackers for each object")
    print("3. Or use object detection to identify different objects")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Vision tracking examples")
    parser.add_argument("--example", type=str, default="basic",
                       choices=["basic", "debug", "compare", "curriculum"],
                       help="Which example to run")

    args = parser.parse_args()

    if args.example == "basic":
        print("Running basic vision tracking example...")
        example_basic_vision_tracking()

    elif args.example == "debug":
        print("Running debug visualization example...")
        example_debug_tracking()

    elif args.example == "compare":
        print("Running tracking method comparison...")
        example_compare_tracking_methods()

    elif args.example == "curriculum":
        print("Running curriculum training example...")
        example_curriculum_training()
