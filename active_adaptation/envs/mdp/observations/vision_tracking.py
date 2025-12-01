"""
Vision-based tracking observations using CoTracker or similar point tracking methods.

These observations use camera RGB images to track object points and estimate object positions.
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple, List
from isaaclab.utils.math import quat_rotate_inverse, matrix_from_quat

from active_adaptation.envs.mdp.base import Observation
from active_adaptation.envs.mdp.trackers.cotracker_wrapper import (
    CoTrackerWrapper,
    TapTrackerWrapper,
)


class VisionTrackedObjectObservation(Observation):
    """
    Base class for vision-based object tracking observations.

    Uses CoTracker to track object points from camera RGB images and
    estimates object position/pose from tracked points.
    """

    def __init__(
        self,
        env,
        camera_name: str = "tiled_camera",
        num_tracking_points: int = 16,
        grid_size: int = 4,
        use_cotracker: bool = True,
        tracker_backend: Optional[str] = None,
        cotracker_checkpoint: Optional[str] = None,
        reinit_on_lost_tracking: bool = True,
        min_visible_points: int = 4,
        use_depth_for_3d: bool = True,
        noise_std: float = 0.0,
        episodic_noise_std: float = 0.0,
    ):
        """
        Initialize vision-based tracking observation.

        Args:
            env: Environment instance
            camera_name: Name of camera sensor in scene
            num_tracking_points: Number of points to track on object
            grid_size: Grid size for automatic point sampling (grid_size x grid_size points)
            use_cotracker: Legacy flag for CoTracker vs. alt backends
            tracker_backend: Explicit backend string ("cotracker" or "tap")
            cotracker_checkpoint: Path to CoTracker checkpoint
            reinit_on_lost_tracking: Re-initialize tracker if too many points are lost
            min_visible_points: Minimum visible points before re-initialization
            use_depth_for_3d: Use depth image to project points to 3D
            noise_std: Standard deviation of per-step noise
            episodic_noise_std: Standard deviation of per-episode noise
        """
        super().__init__(env)

        from isaaclab.sensors import TiledCamera
        self.camera_name = camera_name
        self.camera: TiledCamera = self.env.scene.sensors[camera_name]

        self.num_tracking_points = num_tracking_points
        self.grid_size = grid_size
        self.use_depth_for_3d = use_depth_for_3d
        self.reinit_on_lost_tracking = reinit_on_lost_tracking
        self.min_visible_points = min_visible_points

        # Noise parameters
        self.noise_std = noise_std
        self.episodic_noise_std = episodic_noise_std

        backend = (tracker_backend or ("cotracker" if use_cotracker else "tap")).lower()
        if backend == "cotracker":
            self.tracker = CoTrackerWrapper(
                num_envs=self.num_envs,
                device=self.device,
                model_name="cotracker3_online",
                checkpoint_path=cotracker_checkpoint,
                max_points=num_tracking_points,
            )
        elif backend in {"tap", "tapir"}:
            self.tracker = TapTrackerWrapper(
                num_envs=self.num_envs,
                device=self.device,
                max_points=num_tracking_points,
            )
        else:
            raise ValueError(f"Unsupported tracker backend: {backend}")
        self.tracker_backend = backend

        # Camera intrinsics (will be set in startup)
        self.camera_intrinsics = None
        self.camera_width = self.camera.image_shape[1]
        self.camera_height = self.camera.image_shape[0]

        with torch.device(self.device):
            # Estimated object state from tracking
            self.tracked_object_pos_cam = torch.zeros(self.num_envs, 3, dtype=torch.float32)
            self.tracked_object_pos_w = torch.zeros(self.num_envs, 3, dtype=torch.float32)
            self.tracking_confidence = torch.zeros(self.num_envs, dtype=torch.float32)

            # Noise buffers
            self.episodic_noise = torch.zeros(self.num_envs, 3, dtype=torch.float32)

            # Frame counter for re-initialization
            self.frames_since_init = torch.zeros(self.num_envs, dtype=torch.int32)

    def startup(self):
        """Called once during environment initialization."""
        # Get camera intrinsics
        # Note: Isaac Lab cameras may provide intrinsics via intrinsic_matrices
        if hasattr(self.camera.data, 'intrinsic_matrices'):
            self.camera_intrinsics = self.camera.data.intrinsic_matrices.clone()
        else:
            # Construct from camera parameters
            # Assuming pinhole camera with focal_length
            focal_length = 7.6  # From locomotion.py line 139
            horizontal_aperture = 20.0  # From locomotion.py line 141

            # Compute focal length in pixels
            fx = fy = focal_length * (self.camera_width / horizontal_aperture)
            cx = self.camera_width / 2.0
            cy = self.camera_height / 2.0

            intrinsics = torch.tensor([
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1]
            ], dtype=torch.float32, device=self.device)

            self.camera_intrinsics = intrinsics.unsqueeze(0).expand(self.num_envs, 3, 3)

    def reset(self, env_ids: torch.Tensor):
        """
        Reset tracking for specified environments.

        Initializes point tracking using ground truth object position in the first frame.
        """
        if len(env_ids) == 0:
            return

        # Get RGB image
        rgb_img = self.camera.data.output["rgb"][env_ids]  # [N, H, W, 3]

        # Get initial object bounding box from ground truth (for initialization only)
        object_bbox_2d = self._get_object_bbox_from_ground_truth(env_ids)

        # Initialize tracker with points in the bounding box
        self.tracker.reset(
            env_ids=env_ids,
            rgb_frames=rgb_img,
            query_points=None,  # Will auto-sample from bbox
            object_bbox=object_bbox_2d,
        )

        # Reset noise
        if self.episodic_noise_std > 0:
            self.episodic_noise[env_ids] = torch.randn(
                len(env_ids), 3, device=self.device
            ) * self.episodic_noise_std

        self.frames_since_init[env_ids] = 0

    def update(self, env_ids: torch.Tensor):
        """Update tracking state (called every step)."""
        if len(env_ids) == 0:
            return

        self.frames_since_init[env_ids] += 1

        # Check if re-initialization is needed
        if self.reinit_on_lost_tracking:
            # Count visible points
            visibility = self.tracker.get_visibility(env_ids)
            num_visible = visibility.sum(dim=1)  # [N]

            need_reinit = num_visible < self.min_visible_points
            if need_reinit.any():
                reinit_env_ids = env_ids[need_reinit]
                self.reset(reinit_env_ids)

    def compute(self) -> torch.Tensor:
        """
        Compute vision-based observation.

        Should be implemented by subclasses to return specific observations
        (e.g., object position, tracked points, etc.)
        """
        raise NotImplementedError

    def _get_object_bbox_from_ground_truth(self, env_ids: torch.Tensor) -> torch.Tensor:
        """
        Get object 2D bounding box from ground truth 3D position.

        This is only used for initialization. After that, tracking is purely vision-based.
        """
        # Get object position in world frame (from ground truth)
        # This assumes the command manager has object tracking
        if hasattr(self.env.command_manager, 'object'):
            object_pos_w = self.env.command_manager.object.data.root_link_pos_w[env_ids]
        else:
            # Fallback: use center of image
            N = len(env_ids)
            bbox = torch.zeros(N, 4, device=self.device)
            bbox[:, 0] = self.camera_width * 0.3
            bbox[:, 1] = self.camera_height * 0.3
            bbox[:, 2] = self.camera_width * 0.7
            bbox[:, 3] = self.camera_height * 0.7
            return bbox

        # Project object position to camera frame
        object_pos_2d = self._project_world_to_image(object_pos_w, env_ids)

        # Create bounding box around projected point
        # Assume object is roughly 0.3m x 0.3m in size
        bbox_size_pixels = 50  # pixels (rough estimate)

        bbox = torch.zeros(len(env_ids), 4, device=self.device)
        bbox[:, 0] = torch.clamp(object_pos_2d[:, 0] - bbox_size_pixels, 0, self.camera_width)
        bbox[:, 1] = torch.clamp(object_pos_2d[:, 1] - bbox_size_pixels, 0, self.camera_height)
        bbox[:, 2] = torch.clamp(object_pos_2d[:, 0] + bbox_size_pixels, 0, self.camera_width)
        bbox[:, 3] = torch.clamp(object_pos_2d[:, 1] + bbox_size_pixels, 0, self.camera_height)

        return bbox

    def _project_world_to_image(
        self, points_w: torch.Tensor, env_ids: torch.Tensor
    ) -> torch.Tensor:
        """Project 3D world points to 2D image coordinates."""
        # Get camera pose
        camera_pos_w = self.camera.data.pos_w[env_ids]  # [N, 3]
        camera_quat_w = self.camera.data.quat_w[env_ids]  # [N, 4]

        # Transform points to camera frame
        points_cam = quat_rotate_inverse(camera_quat_w, points_w - camera_pos_w)

        # Project to image plane
        # points_cam: [N, 3] in camera frame (X right, Y down, Z forward)
        K = self.camera_intrinsics[env_ids]  # [N, 3, 3]

        # Homogeneous projection
        points_2d_hom = torch.bmm(K, points_cam.unsqueeze(-1)).squeeze(-1)  # [N, 3]
        points_2d = points_2d_hom[:, :2] / (points_2d_hom[:, 2:3] + 1e-8)  # [N, 2]

        return points_2d

    def _project_image_to_3d(
        self, points_2d: torch.Tensor, depth_img: torch.Tensor, env_ids: torch.Tensor
    ) -> torch.Tensor:
        """
        Project 2D image points to 3D camera frame using depth.

        Args:
            points_2d: 2D points [N, K, 2] in image coordinates
            depth_img: Depth image [N, H, W]
            env_ids: Environment indices [N]

        Returns:
            points_3d_cam: 3D points in camera frame [N, K, 3]
        """
        N, K, _ = points_2d.shape
        H, W = depth_img.shape[-2:]

        # Sample depth at point locations (bilinear interpolation)
        # Normalize coordinates to [-1, 1] for grid_sample
        grid = points_2d.clone()
        grid[..., 0] = 2.0 * grid[..., 0] / W - 1.0  # x
        grid[..., 1] = 2.0 * grid[..., 1] / H - 1.0  # y

        # Sample depth
        depth_values = F.grid_sample(
            depth_img.unsqueeze(1),  # [N, 1, H, W]
            grid.unsqueeze(1),  # [N, 1, K, 2]
            mode='bilinear',
            padding_mode='border',
            align_corners=False
        ).squeeze(1).squeeze(1)  # [N, K]

        # Unproject using camera intrinsics
        K = self.camera_intrinsics[env_ids]  # [N, 3, 3]
        K_inv = torch.inverse(K)  # [N, 3, 3]

        # Homogeneous coordinates
        points_2d_hom = torch.cat([
            points_2d,
            torch.ones(N, K, 1, device=self.device)
        ], dim=-1)  # [N, K, 3]

        # Unproject: P_cam = depth * K_inv @ [u, v, 1]
        points_3d_cam = torch.bmm(
            points_2d_hom.view(N, K, 3),
            K_inv.transpose(1, 2)
        ) * depth_values.unsqueeze(-1)  # [N, K, 3]

        return points_3d_cam

    def _transform_camera_to_world(
        self, points_cam: torch.Tensor, env_ids: torch.Tensor
    ) -> torch.Tensor:
        """Transform points from camera frame to world frame."""
        from isaaclab.utils.math import quat_apply

        camera_pos_w = self.camera.data.pos_w[env_ids]
        camera_quat_w = self.camera.data.quat_w[env_ids]

        # Rotate and translate
        points_w = quat_apply(camera_quat_w, points_cam) + camera_pos_w.unsqueeze(1)

        return points_w

    def _transform_world_to_body(
        self, points_w: torch.Tensor, env_ids: torch.Tensor
    ) -> torch.Tensor:
        """Transform points from world frame to robot body frame."""
        robot_pos_w = self.env.scene.articulations["robot"].data.root_pos_w[env_ids]
        robot_quat_w = self.env.scene.articulations["robot"].data.root_quat_w[env_ids]

        points_b = quat_rotate_inverse(robot_quat_w, points_w - robot_pos_w.unsqueeze(1))

        return points_b


class cotracker_object_position(VisionTrackedObjectObservation):
    """
    Estimates object position using CoTracker point tracking.

    Returns the mean position of tracked points as object position estimate.
    """

    def compute(self) -> torch.Tensor:
        """
        Compute object position from tracked points.

        Returns:
            Object position in robot body frame [num_envs, 3]
        """
        # Get current RGB frame
        rgb_img = self.camera.data.output["rgb"]  # [num_envs, H, W, 3]

        # Update tracking
        tracked_points_2d, visibility = self.tracker.update(rgb_img)  # [N, K, 2], [N, K]

        if self.use_depth_for_3d:
            # Get depth image
            depth_img = self.camera.data.output["distance_to_image_plane"].squeeze(-1)  # [N, H, W]

            # Project to 3D in camera frame
            all_env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.int32)
            tracked_points_3d_cam = self._project_image_to_3d(
                tracked_points_2d, depth_img, all_env_ids
            )  # [N, K, 3]

            # Compute mean position of visible points
            object_pos_cam = torch.zeros(self.num_envs, 3, device=self.device)
            for env_idx in range(self.num_envs):
                visible_mask = visibility[env_idx]
                if visible_mask.any():
                    object_pos_cam[env_idx] = tracked_points_3d_cam[env_idx][visible_mask].mean(dim=0)

            # Transform to world frame
            object_pos_w = self._transform_camera_to_world(
                object_pos_cam.unsqueeze(1), all_env_ids
            ).squeeze(1)

            # Transform to robot body frame
            object_pos_b = self._transform_world_to_body(
                object_pos_w.unsqueeze(1), all_env_ids
            ).squeeze(1)

        else:
            # Use 2D tracking only (compute mean 2D position)
            # This is less useful but can be used for debugging
            mean_2d = self.tracker.compute_mean_position()
            object_pos_b = torch.cat([mean_2d, torch.zeros(self.num_envs, 1, device=self.device)], dim=-1)

        # Apply noise
        noise = torch.randn_like(object_pos_b) * self.noise_std
        object_pos_b = object_pos_b + noise + self.episodic_noise

        return object_pos_b


class cotracker_tracked_points_2d(VisionTrackedObjectObservation):
    """
    Returns raw 2D tracked points from CoTracker.

    Useful for debugging or as direct observation for learning.
    """

    def compute(self) -> torch.Tensor:
        """
        Returns tracked 2D points.

        Returns:
            Tracked points [num_envs, num_points * 2]
        """
        rgb_img = self.camera.data.output["rgb"]
        tracked_points_2d, visibility = self.tracker.update(rgb_img)

        # Flatten points
        points_flat = tracked_points_2d.view(self.num_envs, -1)

        # Normalize to [-1, 1] range
        points_flat[..., 0::2] = 2.0 * points_flat[..., 0::2] / self.camera_width - 1.0
        points_flat[..., 1::2] = 2.0 * points_flat[..., 1::2] / self.camera_height - 1.0

        return points_flat


class cotracker_tracked_points_3d(VisionTrackedObjectObservation):
    """
    Returns 3D tracked points in robot body frame.

    Projects 2D tracked points to 3D using depth and transforms to body frame.
    """

    def compute(self) -> torch.Tensor:
        """
        Returns tracked 3D points in robot body frame.

        Returns:
            Tracked points [num_envs, num_points * 3]
        """
        # Get current RGB and depth
        rgb_img = self.camera.data.output["rgb"]
        depth_img = self.camera.data.output["distance_to_image_plane"].squeeze(-1)

        # Update tracking
        tracked_points_2d, visibility = self.tracker.update(rgb_img)

        # Project to 3D
        all_env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.int32)
        tracked_points_3d_cam = self._project_image_to_3d(
            tracked_points_2d, depth_img, all_env_ids
        )

        # Transform to world frame
        tracked_points_3d_w = self._transform_camera_to_world(
            tracked_points_3d_cam, all_env_ids
        )

        # Transform to robot body frame
        tracked_points_3d_b = self._transform_world_to_body(
            tracked_points_3d_w, all_env_ids
        )

        # Flatten
        points_flat = tracked_points_3d_b.view(self.num_envs, -1)

        # Apply noise
        if self.noise_std > 0:
            noise = torch.randn_like(points_flat) * self.noise_std
            points_flat = points_flat + noise

        return points_flat
