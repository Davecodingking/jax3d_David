"""
Light Probe Grid for spatial lighting features.

A learnable 3D voxel grid that provides spatially-varying lighting features
to augment the texture-based MobileNeRF pipeline.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LightProbeGrid(nn.Module):
    """
    Learnable 3D voxel grid for spatial lighting features.

    The probe stores a grid of learnable feature vectors that are sampled
    using trilinear interpolation based on world-space positions. This allows
    the MLP to learn view-dependent effects that vary spatially.

    Args:
        resolution: Grid resolution (default 16 -> 16^3 voxels)
        num_channels: Number of feature channels per voxel (default 16)
        bounds_min: Minimum corner of probe volume in world space
        bounds_max: Maximum corner of probe volume in world space
        init_std: Standard deviation for random initialization
    """

    def __init__(
        self,
        resolution: int = 16,
        num_channels: int = 16,
        bounds_min: torch.Tensor = None,
        bounds_max: torch.Tensor = None,
        init_std: float = 0.1,
    ):
        super().__init__()

        self.resolution = resolution
        self.num_channels = num_channels

        # Learnable 3D feature grid: (1, C, D, H, W) for grid_sample
        # Using NCDHW format for 3D grid_sample
        self.grid = nn.Parameter(
            torch.randn(1, num_channels, resolution, resolution, resolution) * init_std
        )

        # Register bounds as buffers (not learnable, but saved with model)
        if bounds_min is None:
            bounds_min = torch.tensor([-1.0, -1.0, -1.0])
        if bounds_max is None:
            bounds_max = torch.tensor([1.0, 1.0, 1.0])

        self.register_buffer("bounds_min", bounds_min.float())
        self.register_buffer("bounds_max", bounds_max.float())

    def set_bounds(self, bounds_min: torch.Tensor, bounds_max: torch.Tensor):
        """Update probe bounds (e.g., after loading from preprocessing)."""
        self.bounds_min.copy_(bounds_min.float())
        self.bounds_max.copy_(bounds_max.float())

    def normalize_positions(self, world_pos: torch.Tensor) -> torch.Tensor:
        """
        Normalize world positions to [-1, 1] range for grid_sample.

        Args:
            world_pos: (N, 3) world-space positions

        Returns:
            (N, 3) normalized positions in [-1, 1]
        """
        # Avoid division by zero
        extent = self.bounds_max - self.bounds_min
        extent = torch.clamp(extent, min=1e-6)

        # Normalize to [0, 1] then to [-1, 1]
        normalized = (world_pos - self.bounds_min) / extent
        normalized = normalized * 2.0 - 1.0

        return normalized

    def sample(self, world_pos: torch.Tensor) -> torch.Tensor:
        """
        Sample probe features at given world positions.

        Uses trilinear interpolation via F.grid_sample. Positions outside
        the probe bounds are clamped to the boundary values (border padding).

        Args:
            world_pos: (N, 3) world-space positions

        Returns:
            (N, C) feature vectors sampled from the probe grid
        """
        # Normalize positions to [-1, 1]
        normalized = self.normalize_positions(world_pos)  # (N, 3)

        # grid_sample expects (N, D, H, W, 3) grid for 3D sampling
        # We have (N, 3), reshape to (1, N, 1, 1, 3) for single batch sampling
        # Output will be (1, C, N, 1, 1)
        grid = normalized.view(1, -1, 1, 1, 3)  # (1, N, 1, 1, 3)

        # Sample from the probe grid
        # Input: (1, C, D, H, W), Grid: (1, N, 1, 1, 3)
        # Output: (1, C, N, 1, 1)
        sampled = F.grid_sample(
            self.grid,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )

        # Reshape to (N, C)
        sampled = sampled.squeeze(0).squeeze(-1).squeeze(-1)  # (C, N)
        sampled = sampled.permute(1, 0).contiguous()  # (N, C)

        return sampled

    def sample_with_delta(
        self, world_pos: torch.Tensor, delta_grid: torch.Tensor
    ) -> torch.Tensor:
        """
        Sample probe with per-batch delta modification.

        This method applies a delta to the base probe grid before sampling,
        enabling dynamic/temporal lighting effects where the probe features
        can vary based on frame index or other conditions.

        Args:
            world_pos: (N, 3) world-space positions
            delta_grid: (N, C, R, R, R) delta to add to probe grid per sample

        Returns:
            (N, C) sampled features from modified probe
        """
        n = world_pos.shape[0]

        # Normalize positions to [-1, 1]
        normalized = self.normalize_positions(world_pos)  # (N, 3)

        # Expand base grid to batch: (1, C, R, R, R) -> (N, C, R, R, R)
        grid_batch = self.grid.expand(n, -1, -1, -1, -1)

        # Apply delta in 3D space BEFORE sampling
        modified = grid_batch + delta_grid  # (N, C, R, R, R)

        # grid_sample expects grid in (N, D, H, W, 3) format
        # For per-sample modified grids, we need to sample each point from its
        # corresponding modified grid. Reshape for batched sampling:
        # grid coords: (N, 1, 1, 1, 3) - one sample point per batch item
        grid_coords = normalized.view(n, 1, 1, 1, 3)

        # Sample from modified grids
        # Input: (N, C, D, H, W), Grid: (N, 1, 1, 1, 3)
        # Output: (N, C, 1, 1, 1)
        sampled = F.grid_sample(
            modified,
            grid_coords,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )

        # Reshape to (N, C)
        sampled = sampled.view(n, self.num_channels)

        return sampled

    def sample_with_shared_delta(
        self, world_pos: torch.Tensor, shared_delta: torch.Tensor
    ) -> torch.Tensor:
        """
        Sample probe with a single shared delta for all points.

        This is an optimized version for inference where we use one
        precomputed delta for all pixels in a frame, instead of
        computing per-pixel deltas.

        Args:
            world_pos: (N, 3) world-space positions
            shared_delta: (1, C, R, R, R) single delta grid to apply

        Returns:
            (N, C) sampled features from modified probe
        """
        n = world_pos.shape[0]

        # Normalize positions to [-1, 1]
        normalized = self.normalize_positions(world_pos)  # (N, 3)

        # Apply shared delta to base grid (broadcast)
        modified = self.grid + shared_delta  # (1, C, R, R, R)

        # Sample all points from the same modified grid
        # grid_sample expects grid in (N, D, H, W, 3) format
        # For shared grid, use batch size 1 and sample all points at once
        grid_coords = normalized.view(1, n, 1, 1, 3)  # (1, N, 1, 1, 3)

        # Sample from modified grid
        sampled = F.grid_sample(
            modified,  # (1, C, R, R, R)
            grid_coords,  # (1, N, 1, 1, 3)
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )  # (1, C, N, 1, 1)

        # Reshape to (N, C)
        sampled = sampled.squeeze(0).squeeze(-1).squeeze(-1)  # (C, N)
        sampled = sampled.permute(1, 0).contiguous()  # (N, C)

        return sampled

    def forward(self, world_pos: torch.Tensor) -> torch.Tensor:
        """Forward pass - sample features at world positions."""
        return self.sample(world_pos)

    def export_to_dict(self) -> dict:
        """
        Export probe data for viewer/inference.

        Returns:
            Dictionary containing grid data and bounds for JSON serialization
        """
        return {
            "resolution": self.resolution,
            "num_channels": self.num_channels,
            "bounds_min": self.bounds_min.cpu().tolist(),
            "bounds_max": self.bounds_max.cpu().tolist(),
            # Grid stored as flat list for JSON compatibility
            # Shape: (C, D, H, W) after removing batch dim
            "grid": self.grid.detach().cpu().squeeze(0).numpy().tolist(),
        }
