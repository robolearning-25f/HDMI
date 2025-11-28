"""
Vision-based command manager extension for HDMI tasks.

Extends RobotObjectTracking to optionally use vision-based object position tracking
instead of ground truth simulation data.
"""

import torch
from typing import Optional

from active_adaptation.envs.mdp.commands.hdmi.command import RobotObjectTracking


class RobotObjectVisionTracking(RobotObjectTracking):
    """
    Extends RobotObjectTracking to support vision-based object tracking.

    When use_vision_tracking is True, object positions are estimated from
    camera images using point tracking instead of using ground truth from simulation.
    """

    def __init__(
        self,
        use_vision_tracking: bool = False,
        vision_observation_name: str = "cotracker_object_position",
        vision_tracking_delay: int = 0,
        blend_vision_ground_truth: bool = False,
        vision_weight: float = 1.0,
        **kwargs
    ):
        """
        Initialize vision-based tracking command manager.

        Args:
            use_vision_tracking: If True, use vision-based tracking for object position
            vision_observation_name: Name of the vision tracking observation in observation manager
            vision_tracking_delay: Number of frames to delay vision tracking (simulate processing time)
            blend_vision_ground_truth: If True, blend vision and ground truth positions
            vision_weight: Weight for vision position when blending (0=ground truth, 1=vision only)
            **kwargs: Arguments passed to RobotObjectTracking
        """

        self.use_vision_tracking = use_vision_tracking
        self.vision_observation_name = vision_observation_name
        self.vision_tracking_delay = vision_tracking_delay
        self.blend_vision_ground_truth = blend_vision_ground_truth
        self.vision_weight = vision_weight

        # Vision tracker reference (will be set during environment initialization)
        self.vision_tracker = None

        super().__init__(**kwargs)

        # Vision-based object state
        with torch.device(self.device):
            self.vision_object_pos_w = torch.zeros(self.num_envs, 3, dtype=torch.float32)
            self.vision_object_quat_w = torch.zeros(self.num_envs, 4, dtype=torch.float32)
            self.vision_object_quat_w[:, 0] = 1.0  # Identity quaternion

            # Delay buffer for vision tracking
            if vision_tracking_delay > 0:
                self.vision_pos_buffer = torch.zeros(
                    self.num_envs, vision_tracking_delay + 1, 3, dtype=torch.float32
                )
                self.vision_buffer_idx = torch.zeros(self.num_envs, dtype=torch.int32)

    def set_vision_tracker(self, vision_observation):
        """
        Set reference to vision tracking observation.

        This is called by the environment after observations are initialized.

        Args:
            vision_observation: Instance of VisionTrackedObjectObservation
        """
        self.vision_tracker = vision_observation
        print(f"[RobotObjectVisionTracking] Vision tracker set: {type(vision_observation).__name__}")

    def update(self):
        """
        Update command state, optionally using vision-based tracking.

        If use_vision_tracking is True, replaces ground truth object position
        with vision-based estimate.
        """
        # Call parent update to get ground truth state
        super().update()

        # Override with vision-based tracking if enabled
        if self.use_vision_tracking and self.vision_tracker is not None:
            # Get vision-based object position in body frame
            vision_pos_b = self.vision_tracker.tracked_object_pos_cam

            # Transform to world frame
            # Note: vision_tracker stores in camera frame, we need to transform
            # For simplicity, we can get the full 3D position directly from the observation
            # The observation already handles camera->world->body transforms

            # Get robot state
            robot_pos_w = self.asset.data.root_pos_w
            robot_quat_w = self.asset.data.root_quat_w

            # Transform vision position from body frame to world frame
            from isaaclab.utils.math import quat_apply
            vision_pos_w = quat_apply(robot_quat_w, vision_pos_b) + robot_pos_w

            # Store vision-based position
            self.vision_object_pos_w = vision_pos_w

            # Apply delay if configured
            if self.vision_tracking_delay > 0:
                # Store current position in buffer
                self.vision_pos_buffer[
                    torch.arange(self.num_envs),
                    self.vision_buffer_idx
                ] = vision_pos_w

                # Get delayed position
                delayed_idx = (self.vision_buffer_idx - self.vision_tracking_delay) % (self.vision_tracking_delay + 1)
                vision_pos_w = self.vision_pos_buffer[torch.arange(self.num_envs), delayed_idx]

                # Update buffer index
                self.vision_buffer_idx = (self.vision_buffer_idx + 1) % (self.vision_tracking_delay + 1)

            # Blend or replace ground truth position
            if self.blend_vision_ground_truth:
                # Blend vision and ground truth
                self.object_pos_w = (
                    self.vision_weight * vision_pos_w +
                    (1.0 - self.vision_weight) * self.object_pos_w
                )
            else:
                # Fully replace with vision-based position
                self.object_pos_w = vision_pos_w

            # Note: We keep object_quat_w from ground truth since orientation
            # estimation from point tracking is more complex and less reliable
            # This can be extended to estimate orientation using point cloud alignment


class RobotObjectVisionTrackingWithOrientation(RobotObjectVisionTracking):
    """
    Extended vision tracking that also estimates object orientation.

    Uses tracked points to estimate object pose (position + orientation)
    via point cloud alignment methods like ICP or Kabsch algorithm.
    """

    def __init__(
        self,
        estimate_orientation: bool = True,
        orientation_estimation_method: str = "pca",  # "pca", "kabsch", or "none"
        **kwargs
    ):
        """
        Initialize vision tracking with orientation estimation.

        Args:
            estimate_orientation: If True, estimate object orientation from tracked points
            orientation_estimation_method: Method for orientation estimation
                - "pca": Use PCA on tracked points
                - "kabsch": Use Kabsch algorithm if reference points are available
                - "none": Use ground truth orientation
            **kwargs: Arguments passed to RobotObjectVisionTracking
        """
        super().__init__(**kwargs)

        self.estimate_orientation = estimate_orientation
        self.orientation_estimation_method = orientation_estimation_method

    def update(self):
        """Update with orientation estimation."""
        super().update()

        if self.estimate_orientation and self.vision_tracker is not None:
            if self.orientation_estimation_method == "pca":
                self._estimate_orientation_pca()
            elif self.orientation_estimation_method == "kabsch":
                self._estimate_orientation_kabsch()
            # else: use ground truth orientation (default behavior)

    def _estimate_orientation_pca(self):
        """Estimate object orientation using PCA of tracked points."""
        # Get tracked 3D points in world frame
        if not hasattr(self.vision_tracker, 'tracked_points_3d_w'):
            return

        tracked_points_w = self.vision_tracker.tracked_points_3d_w  # [num_envs, K, 3]
        visibility = self.vision_tracker.tracker.get_visibility()  # [num_envs, K]

        for env_idx in range(self.num_envs):
            visible_mask = visibility[env_idx]
            if visible_mask.sum() < 3:
                # Need at least 3 points for PCA
                continue

            visible_points = tracked_points_w[env_idx][visible_mask]  # [N, 3]

            # Center points
            centered_points = visible_points - visible_points.mean(dim=0)

            # Compute covariance matrix
            cov = (centered_points.T @ centered_points) / len(centered_points)

            # Eigen decomposition
            try:
                eigenvalues, eigenvectors = torch.linalg.eigh(cov)

                # Sort by eigenvalues (descending)
                idx = eigenvalues.argsort(descending=True)
                eigenvectors = eigenvectors[:, idx]

                # Ensure right-handed coordinate system
                if torch.det(eigenvectors) < 0:
                    eigenvectors[:, 2] *= -1

                # Convert rotation matrix to quaternion
                from isaaclab.utils.math import quat_from_matrix
                quat = quat_from_matrix(eigenvectors)

                # Update orientation (blend with ground truth if configured)
                if self.blend_vision_ground_truth:
                    # Quaternion blending is complex, for now just use vision
                    self.vision_object_quat_w[env_idx] = quat
                    # Could use SLERP for proper quaternion interpolation
                else:
                    self.object_quat_w[env_idx] = quat

            except:
                # If eigendecomposition fails, keep ground truth
                pass

    def _estimate_orientation_kabsch(self):
        """
        Estimate object orientation using Kabsch algorithm.

        This aligns tracked points with reference object points from the motion dataset.
        """
        # Get tracked 3D points
        if not hasattr(self.vision_tracker, 'tracked_points_3d_w'):
            return

        tracked_points_w = self.vision_tracker.tracked_points_3d_w  # [num_envs, K, 3]
        visibility = self.vision_tracker.tracker.get_visibility()  # [num_envs, K]

        # Get reference object points from motion dataset
        # This requires knowing which points on the object are being tracked
        # For now, this is a placeholder - you'd need to store reference points

        # Kabsch algorithm:
        # 1. Center both point sets
        # 2. Compute cross-covariance matrix H = P.T @ Q
        # 3. SVD: H = U S V.T
        # 4. Rotation: R = V U.T
        # 5. Convert R to quaternion

        # Implementation would go here
        pass
