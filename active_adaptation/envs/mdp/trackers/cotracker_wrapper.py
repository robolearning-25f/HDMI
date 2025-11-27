"""
CoTracker3 Online wrapper for point tracking in multi-environment settings.

This module provides a wrapper around CoTracker3 Online API to handle
batch processing across multiple parallel Isaac Lab environments.
"""

import torch
import torch.nn.functional as F
from typing import Optional, Tuple, List
import numpy as np


class CoTrackerWrapper:
    """
    Wrapper for CoTracker3 Online point tracking model.

    Uses the CoTracker3 online API for efficient streaming point tracking.
    Handles multiple parallel environments with per-environment tracking state.
    """

    def __init__(
        self,
        num_envs: int,
        device: torch.device,
        model_name: str = "cotracker3_online",
        checkpoint_path: Optional[str] = None,
        grid_size: int = 10,
        max_points: int = 100,
        window_len: int = 8,
    ):
        """
        Initialize CoTracker3 Online wrapper.

        Args:
            num_envs: Number of parallel environments
            device: Device to run tracking on
            model_name: CoTracker model variant ('cotracker3_online' recommended)
            checkpoint_path: Path to model checkpoint (if None, uses torch.hub)
            grid_size: Grid size for automatic point selection
            max_points: Maximum number of points to track
            window_len: Window length for online tracking (default: 8 frames)
        """
        self.num_envs = num_envs
        self.device = device
        self.model_name = model_name
        self.grid_size = grid_size
        self.max_points = max_points
        self.window_len = window_len

        # Initialize CoTracker model
        self._init_model(checkpoint_path)

        # Tracking state for each environment
        with torch.device(device):
            self.tracked_points = torch.zeros(num_envs, max_points, 2, dtype=torch.float32)
            self.tracked_visibility = torch.zeros(num_envs, max_points, dtype=torch.bool)
            self.num_tracked_points = torch.zeros(num_envs, dtype=torch.int32)

            # Frame buffer for online tracking (stores recent frames)
            self.frame_buffer = []  # List of [num_envs, 3, H, W] tensors
            self.max_buffer_len = window_len * 2

            self.is_initialized = torch.zeros(num_envs, dtype=torch.bool)
            self.is_first_step = torch.ones(num_envs, dtype=torch.bool)

            # Store query points for each environment
            self.query_points = torch.zeros(num_envs, max_points, 2, dtype=torch.float32)

    def _init_model(self, checkpoint_path: Optional[str] = None):
        """Initialize the CoTracker3 Online model."""
        try:
            # Load CoTracker3 Online from torch.hub
            print(f"[CoTrackerWrapper] Loading {self.model_name} from torch.hub...")

            if checkpoint_path is not None:
                # Load from local checkpoint
                self.model = torch.hub.load(
                    "facebookresearch/co-tracker",
                    self.model_name,
                    checkpoint=checkpoint_path
                )
            else:
                # Load default pretrained model
                self.model = torch.hub.load(
                    "facebookresearch/co-tracker",
                    self.model_name
                )

            self.model = self.model.to(self.device)
            self.model.eval()
            self.use_cotracker = True

            # Get model's step size
            self.step_size = getattr(self.model, 'step', 4)

            print(f"[CoTrackerWrapper] ✓ {self.model_name} loaded successfully")
            print(f"[CoTrackerWrapper]   - Step size: {self.step_size}")
            print(f"[CoTrackerWrapper]   - Device: {self.device}")

        except Exception as e:
            print(f"[CoTrackerWrapper] Failed to load CoTracker: {e}")
            print("[CoTrackerWrapper] Falling back to simple optical flow tracking.")
            self.use_cotracker = False
            self.step_size = 1
            self._init_simple_tracker()

    def _init_simple_tracker(self):
        """Initialize a simple template matching tracker as fallback."""
        self.use_cotracker = False
        self.model = None
        print("[CoTrackerWrapper] Simple tracker initialized (fallback mode)")

    def reset(
        self,
        env_ids: torch.Tensor,
        rgb_frames: torch.Tensor,
        query_points: Optional[torch.Tensor] = None,
        object_bbox: Optional[torch.Tensor] = None,
    ):
        """
        Reset tracker for specified environments.

        Args:
            env_ids: Environment indices to reset [N]
            rgb_frames: RGB images [N, H, W, 3] or [N, 3, H, W], values in [0, 255]
            query_points: Initial points to track [N, K, 2] (optional)
            object_bbox: Object bounding boxes [N, 4] as (x1, y1, x2, y2) (optional)
        """
        N = len(env_ids)

        # Convert RGB format if needed
        if rgb_frames.ndim == 4 and rgb_frames.shape[-1] == 3:
            # [N, H, W, 3] -> [N, 3, H, W]
            rgb_frames = rgb_frames.permute(0, 3, 1, 2)

        # Normalize to [0, 1]
        if rgb_frames.max() > 1.0:
            rgb_frames = rgb_frames.float() / 255.0

        H, W = rgb_frames.shape[-2:]

        # Auto-generate query points if not provided
        if query_points is None:
            if object_bbox is not None:
                # Sample points within object bounding box
                query_points = self._sample_points_from_bbox(object_bbox, self.grid_size)
            else:
                # Sample points uniformly across image
                query_points = self._sample_grid_points(N, H, W, self.grid_size)

        num_points = query_points.shape[1]

        # Store query points and initial state
        for i, env_idx in enumerate(env_ids):
            self.query_points[env_idx, :num_points] = query_points[i]
            self.tracked_points[env_idx, :num_points] = query_points[i]
            self.tracked_visibility[env_idx, :num_points] = True
            self.num_tracked_points[env_idx] = num_points
            self.is_initialized[env_idx] = True
            self.is_first_step[env_idx] = True

        # Clear frame buffer for reset environments
        # Note: We manage buffers per-environment, so we don't need to clear here
        # The is_first_step flag handles CoTracker initialization

        print(f"[CoTrackerWrapper] Reset {len(env_ids)} environments with {num_points} points each")

    def update(self, rgb_frames: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update point tracking with new frames using CoTracker3 Online API.

        Args:
            rgb_frames: RGB images [num_envs, H, W, 3] or [num_envs, 3, H, W]

        Returns:
            tracked_points: Tracked point coordinates [num_envs, max_points, 2]
            visibility: Visibility flags [num_envs, max_points]
        """
        # Convert RGB format if needed
        if rgb_frames.ndim == 4 and rgb_frames.shape[-1] == 3:
            rgb_frames = rgb_frames.permute(0, 3, 1, 2)

        # Normalize to [0, 1]
        if rgb_frames.max() > 1.0:
            rgb_frames = rgb_frames.float() / 255.0

        # Add to frame buffer
        self.frame_buffer.append(rgb_frames.clone())
        if len(self.frame_buffer) > self.max_buffer_len:
            self.frame_buffer.pop(0)

        if self.use_cotracker:
            self._track_with_cotracker_online(rgb_frames)
        else:
            self._track_with_simple_method(rgb_frames)

        return self.tracked_points.clone(), self.tracked_visibility.clone()

    def _track_with_cotracker_online(self, current_frame: torch.Tensor):
        """
        Track points using CoTracker3 Online API.

        CoTracker3 Online processes frames incrementally with overlapping windows.

        Args:
            current_frame: Current RGB frame (already added to frame_buffer)
        """
        # Process each environment independently
        # TODO: Optimize to batch multiple environments if possible

        for env_idx in range(self.num_envs):
            if not self.is_initialized[env_idx]:
                continue

            num_points = self.num_tracked_points[env_idx]
            if num_points == 0:
                continue

            try:
                # Prepare video chunk: need at least 2 frames for tracking
                if len(self.frame_buffer) < 2:
                    # Not enough frames yet, keep initial points
                    continue

                # For first step, initialize tracker with initial frame
                if self.is_first_step[env_idx]:
                    # Get initial frame
                    initial_frame = self.frame_buffer[0][env_idx:env_idx+1]  # [1, 3, H, W]

                    # Prepare as video [B=1, T=1, C, H, W]
                    video_init = initial_frame.unsqueeze(1)  # [1, 1, 3, H, W]

                    # Initialize CoTracker3 Online with grid_size
                    # This sets up internal state
                    with torch.no_grad():
                        self.model(
                            video_chunk=video_init,
                            is_first_step=True,
                            grid_size=self.grid_size,
                        )

                    self.is_first_step[env_idx] = False

                    # On first step, just keep initial points
                    continue

                # For subsequent steps, process with overlapping windows
                # Get recent frames for this environment
                buffer_len = min(len(self.frame_buffer), self.step_size * 2)

                # Stack recent frames into video chunk
                recent_frames = torch.stack([
                    self.frame_buffer[i][env_idx:env_idx+1]  # [1, 3, H, W]
                    for i in range(-buffer_len, 0)
                ], dim=1)  # [1, T, 3, H, W]

                # Run CoTracker3 Online
                with torch.no_grad():
                    pred_tracks, pred_visibility = self.model(
                        video_chunk=recent_frames
                    )
                    # pred_tracks: [1, T, N, 2] - tracks for all frames in window
                    # pred_visibility: [1, T, N, 1] - visibility for all frames

                # Extract predictions for the last (current) frame
                last_frame_tracks = pred_tracks[0, -1, :num_points, :]  # [N, 2]
                last_frame_vis = pred_visibility[0, -1, :num_points, 0]  # [N]

                # Update tracked state
                self.tracked_points[env_idx, :num_points] = last_frame_tracks
                self.tracked_visibility[env_idx, :num_points] = last_frame_vis > 0.5

            except Exception as e:
                print(f"[CoTrackerWrapper] Tracking failed for env {env_idx}: {e}")
                # Keep previous predictions on error
                continue

    def _track_with_simple_method(self, current_frame: torch.Tensor):
        """Simple tracking fallback using template matching."""
        if len(self.frame_buffer) < 2:
            return

        prev_frame = self.frame_buffer[-2]
        curr_frame = current_frame

        for env_idx in range(self.num_envs):
            if not self.is_initialized[env_idx]:
                continue

            num_points = self.num_tracked_points[env_idx]
            if num_points == 0:
                continue

            # Convert to grayscale
            prev_gray = prev_frame[env_idx].mean(dim=0, keepdim=True)  # [1, H, W]
            curr_gray = curr_frame[env_idx].mean(dim=0, keepdim=True)  # [1, H, W]

            # Track each point with template matching
            for pt_idx in range(num_points):
                if not self.tracked_visibility[env_idx, pt_idx]:
                    continue

                pt = self.tracked_points[env_idx, pt_idx]
                new_pt = self._track_point_template_matching(
                    prev_gray, curr_gray, pt, window_size=15, search_size=25
                )

                if new_pt is not None:
                    self.tracked_points[env_idx, pt_idx] = new_pt
                else:
                    self.tracked_visibility[env_idx, pt_idx] = False

    def _track_point_template_matching(
        self,
        prev_img: torch.Tensor,
        curr_img: torch.Tensor,
        point: torch.Tensor,
        window_size: int = 15,
        search_size: int = 25,
    ) -> Optional[torch.Tensor]:
        """Track a single point using template matching."""
        x, y = point.int()
        H, W = prev_img.shape[-2:]

        half_win = window_size // 2
        half_search = search_size // 2

        # Extract template from previous frame
        x1_t, y1_t = max(0, x - half_win), max(0, y - half_win)
        x2_t, y2_t = min(W, x + half_win + 1), min(H, y + half_win + 1)

        if x2_t - x1_t < 5 or y2_t - y1_t < 5:
            return None

        template = prev_img[..., y1_t:y2_t, x1_t:x2_t]

        # Define search region in current frame
        x1_s, y1_s = max(0, x - half_search), max(0, y - half_search)
        x2_s, y2_s = min(W, x + half_search + 1), min(H, y + half_search + 1)

        if x2_s - x1_s < window_size or y2_s - y1_s < window_size:
            return None

        search_region = curr_img[..., y1_s:y2_s, x1_s:x2_s]

        # Compute normalized cross-correlation
        try:
            template_norm = (template - template.mean()) / (template.std() + 1e-8)
            search_norm = (search_region - search_region.mean()) / (search_region.std() + 1e-8)

            # Simple correlation
            corr = F.conv2d(
                search_norm.unsqueeze(0),
                template_norm.unsqueeze(0).unsqueeze(0),
                padding=0
            )[0, 0]

            # Find maximum correlation
            max_idx = corr.argmax()
            max_y, max_x = max_idx // corr.shape[1], max_idx % corr.shape[1]

            # Convert to absolute coordinates
            new_x = x1_s + max_x + half_win
            new_y = y1_s + max_y + half_win

            return torch.tensor([new_x, new_y], dtype=torch.float32, device=point.device)

        except:
            return None

    def _sample_grid_points(
        self, batch_size: int, H: int, W: int, grid_size: int
    ) -> torch.Tensor:
        """Sample points uniformly in a grid."""
        x = torch.linspace(W * 0.1, W * 0.9, grid_size, device=self.device)
        y = torch.linspace(H * 0.1, H * 0.9, grid_size, device=self.device)

        grid_x, grid_y = torch.meshgrid(x, y, indexing='xy')
        points = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1)  # [grid_size^2, 2]

        # Repeat for batch
        points = points.unsqueeze(0).expand(batch_size, -1, -1)  # [B, K, 2]

        return points

    def _sample_points_from_bbox(
        self, bbox: torch.Tensor, grid_size: int
    ) -> torch.Tensor:
        """Sample points within bounding boxes."""
        # bbox: [N, 4] as (x1, y1, x2, y2)
        N = bbox.shape[0]

        x = torch.linspace(0.1, 0.9, grid_size, device=self.device)
        y = torch.linspace(0.1, 0.9, grid_size, device=self.device)

        grid_x, grid_y = torch.meshgrid(x, y, indexing='xy')
        grid_points = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=-1)  # [K, 2]

        # Scale to bounding box
        points = torch.zeros(N, grid_size * grid_size, 2, device=self.device)
        for i in range(N):
            x1, y1, x2, y2 = bbox[i]
            w, h = x2 - x1, y2 - y1
            points[i, :, 0] = x1 + grid_points[:, 0] * w
            points[i, :, 1] = y1 + grid_points[:, 1] * h

        return points

    def get_tracked_points(self, env_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Get current tracked points for specified environments."""
        if env_ids is None:
            return self.tracked_points.clone()
        else:
            return self.tracked_points[env_ids].clone()

    def get_visibility(self, env_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Get visibility flags for tracked points."""
        if env_ids is None:
            return self.tracked_visibility.clone()
        else:
            return self.tracked_visibility[env_ids].clone()

    def compute_mean_position(self, env_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute mean position of visible tracked points."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)

        mean_pos = torch.zeros(len(env_ids), 2, device=self.device)

        for i, env_idx in enumerate(env_ids):
            visible_mask = self.tracked_visibility[env_idx, :self.num_tracked_points[env_idx]]
            if visible_mask.any():
                visible_points = self.tracked_points[env_idx, :self.num_tracked_points[env_idx]][visible_mask]
                mean_pos[i] = visible_points.mean(dim=0)

        return mean_pos
