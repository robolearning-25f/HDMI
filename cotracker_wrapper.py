import torch
from typing import Optional, Tuple

class CoTrackerWrapper:
    def __init__(
        self,
        num_envs: int,
        device: torch.device,
        model_name: str = "cotracker3_online",
        checkpoint_path: Optional[str] = None,
        max_points: int = 100,
        window_len: int = 8,
    ):
        self.num_envs = num_envs
        self.device = device
        self.model_name = model_name
        self.max_points = max_points
        self.window_len = window_len

        # Initialize CoTracker model
        self._init_model(checkpoint_path)

        # Tracking state for each environment
        with torch.device(device):
            self.tracked_points = torch.zeros(num_envs, max_points, 2, dtype=torch.float32)
            self.tracked_visibility = torch.zeros(num_envs, max_points, dtype=torch.bool)
            self.num_tracked_points = torch.zeros(num_envs, dtype=torch.int32)

            # Frame buffer for sliding window tracking (per environment)
            self.frame_buffers = [[] for _ in range(num_envs)]  # Each env has its own list of frames

            self.is_initialized = torch.zeros(num_envs, dtype=torch.bool)

            # Store query points for each environment (as float)
            self.query_points = torch.zeros(num_envs, max_points, 2, dtype=torch.float32)

        # Store per-frame trajectories for each environment
        self.track_history = [[] for _ in range(num_envs)]
        self.visibility_history = [[] for _ in range(num_envs)]

    def _init_model(self, checkpoint_path: Optional[str] = None):
        """Initialize the CoTracker model."""
        # Load CoTracker from torch.hub
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

        # Get model's window/step configuration so our buffers align with it
        self.window_len = self.model.model.window_len
        self.step_len = self.model.step

        print(f"[CoTrackerWrapper] ✓ {self.model_name} loaded successfully")
        print(f"[CoTrackerWrapper]   - Tracking mode: query-based online")
        print(f"[CoTrackerWrapper]   - Device: {self.device}")
        print(f"[CoTrackerWrapper]   - Window length: {self.window_len}")


    def reset(
        self,
        env_ids: torch.Tensor,
        rgb_frames: torch.Tensor,
        query_points: torch.Tensor,
    ):
        """
        Reset tracker for specified environments.

        Args:
            env_ids: Environment indices to reset [N]
            rgb_frames: RGB images [N, H, W, 3] or [N, 3, H, W], values in [0, 255]
            query_points: Initial points to track [N, K, 2] as float
        """
        # Convert RGB format if needed
        if rgb_frames.ndim == 4 and rgb_frames.shape[-1] == 3:
            rgb_frames = rgb_frames.permute(0, 3, 1, 2)
        # Normalize to [0, 1]
        if rgb_frames.max() > 1.0:
            rgb_frames = rgb_frames.float() / 255.0

        # Ensure query_points are float32 for CoTracker queries API
        query_points = query_points.float()

        num_points = query_points.shape[1]
        
        track_history = []
        # Store query points and initial state
        for i, env_idx in enumerate(env_ids):
            self.query_points[env_idx, :num_points] = query_points[i]
            self.tracked_points[env_idx, :num_points] = query_points[i]
            self.tracked_visibility[env_idx, :num_points] = True
            self.num_tracked_points[env_idx] = num_points
            self.is_initialized[env_idx] = True
            # Clear frame buffer for this environment and add initial frame
            self.frame_buffers[env_idx] = [rgb_frames[env_idx].clone() for _ in range(self.step_len)]  # [1, 3, H, W]
            track_history.append(self.tracked_points[env_idx].clone())

        video = torch.stack([torch.stack(row, dim=0) for row in self.frame_buffers], dim=0)
        query = torch.stack(track_history, dim=0)

        queries = torch.zeros(self.num_envs, num_points, 3, dtype=torch.float32, device=self.device)
        queries[..., 1:] = query
        # Run CoTracker on the window
        with torch.no_grad():
            self.model(
                video_chunk=video,
                is_first_step=True,  # Treat each window as independent
                queries=queries
            )
        self.current_frame = video
        print(f"[CoTrackerWrapper] Reset {len(env_ids)} environments with {num_points} points each")

    def update(self, rgb_frames: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update point tracking with new frames using sliding window tracking.

        Args:
            rgb_frames: RGB images [num_envs, H, W, 3] or [num_envs, 3, H, W]

        Returns:
            tracked_points: Trajectory history [num_envs, num_frames, max_points, 2]
            visibility: Visibility history [num_envs, num_frames, max_points]
        """
        # Convert RGB format if needed
        if rgb_frames.ndim == 4 and rgb_frames.shape[-1] == 3:
            rgb_frames = rgb_frames.permute(0, 3, 1, 2)

        # Normalize to [0, 1]
        if rgb_frames.max() > 1.0:
            rgb_frames = rgb_frames.float() / 255.0

        # Add new frame to each environment's buffer
        for env_idx in range(self.num_envs):
            self.frame_buffers[env_idx].append(rgb_frames[env_idx].clone())  # [1, 3, H, W]

        # Track with sliding window
        self._track_with_sliding_window(rgb_frames)

        # Append current state to history
        self._append_history_entries()

        return self._build_history_tensors()

    def _track_with_sliding_window(self, rgb_frames: torch.Tensor):
        """
        Track points using sliding window approach with CoTracker.

        For each environment:
        - Wait until we have >= window_len frames
        - Use the last window_len frames for tracking
        - Use current tracked points as query points (for continuity)
        - Extract only the newest frame's tracking result
        """
        # Process each environment independently
        for env_idx in range(self.num_envs):
            if not self.is_initialized[env_idx]:
                continue

            num_points = self.num_tracked_points[env_idx]
            if num_points == 0:
                continue

            # Add new frame to each environment's buffer
            forward_frames = rgb_frames.repeat(1, self.step_len, 1, 1, 1)
            forward_chunk = torch.cat([self.current_frame, forward_frames], dim=1)
            # Run CoTracker on the window
            with torch.no_grad():
                pred_tracks, pred_visibility = self.model(
                    video_chunk=forward_chunk,
                )
            self.current_frame = forward_frames
            # pred_tracks: [1, T, N, 2] - tracks for all frames in window
            # pred_visibility: [1, T, N, 1] - visibility for all frames

            # Extract predictions for the last (newest) frame only
            last_frame_tracks = pred_tracks[:, -1, :num_points, :]  # [B, N, 2]
            last_frame_vis = pred_visibility[:, -1, :num_points]  # [B, N]

            # Update tracked state with newest frame result
            self.tracked_points[:, :num_points] = last_frame_tracks
            self.tracked_visibility[:, :num_points] = last_frame_vis > 0.5


    def _append_history_entries(self):
        for env_idx in range(self.num_envs):
            if not self.is_initialized[env_idx]:
                continue
            num_points = self.num_tracked_points[env_idx]
            if num_points == 0:
                continue
            self.track_history[env_idx].append(self.tracked_points[env_idx].clone())
            self.visibility_history[env_idx].append(self.tracked_visibility[env_idx].clone())

    def _build_history_tensors(self) -> Tuple[torch.Tensor, torch.Tensor]:
        max_history = max((len(hist) for hist in self.track_history), default=0)
        if max_history == 0:
            empty_tracks = torch.zeros(
                self.num_envs, 0, self.max_points, 2, dtype=torch.float32, device=self.device
            )
            empty_visibility = torch.zeros(
                self.num_envs, 0, self.max_points, dtype=torch.bool, device=self.device
            )
            return empty_tracks, empty_visibility

        tracks = torch.zeros(
            self.num_envs, max_history, self.max_points, 2, dtype=torch.float32, device=self.device
        )
        visibility = torch.zeros(
            self.num_envs, max_history, self.max_points, dtype=torch.bool, device=self.device
        )
        for env_idx, env_hist in enumerate(self.track_history):
            if not env_hist:
                continue
            stacked_tracks = torch.stack(env_hist)
            stacked_visibility = torch.stack(self.visibility_history[env_idx])
            hist_len = stacked_tracks.shape[0]
            tracks[env_idx, :hist_len] = stacked_tracks
            visibility[env_idx, :hist_len] = stacked_visibility
        return tracks, visibility


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

class SingleEnvCoTrackerWrapper:
    """
    Single environment version of CoTrackerWrapper.
    Simplified for use with observation classes that track one environment at a time.
    """

    def __init__(
        self,
        device: torch.device,
        model_name: str = "cotracker3_online",
        checkpoint_path: Optional[str] = None,
        max_points: int = 100,
    ):
        self.device = device
        self.model_name = model_name
        self.max_points = max_points

        # Initialize CoTracker model
        self._init_model(checkpoint_path)

        # Tracking state (single environment)
        with torch.device(device):
            self.tracked_points = torch.zeros(max_points, 2, dtype=torch.float32)
            self.tracked_visibility = torch.zeros(max_points, dtype=torch.bool)
            self.num_tracked_points = 0

            # Store query points (as float)
            self.query_points = torch.zeros(max_points, 2, dtype=torch.float32)

            self.is_initialized = False

        # Store per-frame trajectories
        self.track_history = []
        self.visibility_history = []

        # Current frame buffer for sliding window
        self.current_frame = None

    def _init_model(self, checkpoint_path: Optional[str] = None):
        """Initialize the CoTracker model."""
        # Load CoTracker from torch.hub
        print(f"[SingleEnvCoTrackerWrapper] Loading {self.model_name} from torch.hub...")

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

        # Get model's window/step configuration
        self.window_len = self.model.model.window_len
        self.step_len = self.model.step

        print(f"[SingleEnvCoTrackerWrapper] ✓ {self.model_name} loaded successfully")
        print(f"[SingleEnvCoTrackerWrapper]   - Tracking mode: query-based online")
        print(f"[SingleEnvCoTrackerWrapper]   - Device: {self.device}")
        print(f"[SingleEnvCoTrackerWrapper]   - Window length: {self.window_len}")
        print(f"[SingleEnvCoTrackerWrapper]   - Step length: {self.step_len}")

    def reset(
        self,
        rgb_frame: torch.Tensor,
        query_points: torch.Tensor,
    ):
        """
        Reset tracker with initial frame and query points.

        Args:
            rgb_frame: RGB image [H, W, 3] or [3, H, W], values in [0, 255]
            query_points: Initial points to track [K, 2] as float (x, y coordinates)
        """
        # Convert RGB format if needed
        if rgb_frame.ndim == 3 and rgb_frame.shape[-1] == 3:
            rgb_frame = rgb_frame.permute(2, 0, 1)  # [3, H, W]

        # Add batch dimension if needed
        if rgb_frame.ndim == 3:
            rgb_frame = rgb_frame.unsqueeze(0)  # [1, 3, H, W]

        # Normalize to [0, 1]
        if rgb_frame.max() > 1.0:
            rgb_frame = rgb_frame.float() / 255.0

        # Ensure query_points are float32
        query_points = query_points.float()
        num_points = query_points.shape[0]

        # Store query points and initial state
        self.query_points[:num_points] = query_points
        self.tracked_points[:num_points] = query_points
        self.tracked_visibility[:num_points] = True
        self.num_tracked_points = num_points
        self.is_initialized = True

        # Initialize frame buffer with repeated initial frame
        # Shape: [1, step_len, 3, H, W]
        video = rgb_frame.repeat(1, self.step_len, 1, 1, 1)

        # Prepare queries: [1, K, 3] where first dimension is time (always 0 for initial frame)
        queries = torch.zeros(1, num_points, 3, dtype=torch.float32, device=self.device)
        queries[0, :, 1:] = query_points  # queries[:, 0] = 0 (first frame), queries[:, 1:] = (x, y)

        # Run CoTracker on the initial window
        with torch.no_grad():
            self.model(
                video_chunk=video,
                is_first_step=True,
                queries=queries
            )

        self.current_frame = video

        # Clear history
        self.track_history = []
        self.visibility_history = []

        print(f"[SingleEnvCoTrackerWrapper] Reset with {num_points} points")

    def update(self, rgb_frame: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update point tracking with new frame.

        Args:
            rgb_frame: RGB image [H, W, 3] or [3, H, W]

        Returns:
            tracked_points: Trajectory history [num_frames, max_points, 2]
            visibility: Visibility history [num_frames, max_points]
        """
        if not self.is_initialized:
            raise RuntimeError("Tracker not initialized. Call reset() first.")

        # Convert RGB format if needed
        if rgb_frame.ndim == 3 and rgb_frame.shape[-1] == 3:
            rgb_frame = rgb_frame.permute(2, 0, 1)  # [3, H, W]

        # Add batch dimension if needed
        if rgb_frame.ndim == 3:
            rgb_frame = rgb_frame.unsqueeze(0)  # [1, 3, H, W]

        # Normalize to [0, 1]
        if rgb_frame.max() > 1.0:
            rgb_frame = rgb_frame.float() / 255.0

        # Track with sliding window
        self._track_with_sliding_window(rgb_frame)

        # Append current state to history
        self._append_history_entry()

        return self._build_history_tensors()

    def _track_with_sliding_window(self, rgb_frame: torch.Tensor):
        """
        Track points using sliding window approach with CoTracker.

        Args:
            rgb_frame: RGB frame [1, 3, H, W]
        """
        if self.num_tracked_points == 0:
            return

        # Create forward chunk: current buffer + step_len copies of new frame
        # Shape: [1, step_len, 3, H, W]
        forward_frames = rgb_frame.repeat(1, self.step_len, 1, 1, 1)

        # Concatenate with current buffer: [1, window_len + step_len, 3, H, W]
        forward_chunk = torch.cat([self.current_frame, forward_frames], dim=1)

        # Run CoTracker on the window
        with torch.no_grad():
            pred_tracks, pred_visibility = self.model(
                video_chunk=forward_chunk,
            )

        # Update current frame buffer to the new frames
        self.current_frame = forward_frames

        # pred_tracks: [1, T, N, 2] - tracks for all frames in window
        # pred_visibility: [1, T, N] - visibility for all frames

        # Extract predictions for the last (newest) frame only
        last_frame_tracks = pred_tracks[0, -1, :self.num_tracked_points, :]  # [N, 2]
        last_frame_vis = pred_visibility[0, -1, :self.num_tracked_points]  # [N]

        # Update tracked state with newest frame result
        self.tracked_points[:self.num_tracked_points] = last_frame_tracks
        self.tracked_visibility[:self.num_tracked_points] = last_frame_vis > 0.5

    def _append_history_entry(self):
        """Append current tracking state to history."""
        if self.num_tracked_points == 0:
            return
        self.track_history.append(self.tracked_points.clone())
        self.visibility_history.append(self.tracked_visibility.clone())

    def _build_history_tensors(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build trajectory tensors from history."""
        if not self.track_history:
            empty_tracks = torch.zeros(
                0, self.max_points, 2, dtype=torch.float32, device=self.device
            )
            empty_visibility = torch.zeros(
                0, self.max_points, dtype=torch.bool, device=self.device
            )
            return empty_tracks, empty_visibility

        # Stack history: [num_frames, max_points, 2/bool]
        tracks = torch.stack(self.track_history)
        visibility = torch.stack(self.visibility_history)

        return tracks, visibility

    def get_current_points(self) -> torch.Tensor:
        """Get current tracked points [num_points, 2]."""
        return self.tracked_points[:self.num_tracked_points].clone()

    def get_current_visibility(self) -> torch.Tensor:
        """Get current visibility flags [num_points]."""
        return self.tracked_visibility[:self.num_tracked_points].clone()
