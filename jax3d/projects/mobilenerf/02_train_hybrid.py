import os
import json
import argparse
import math

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


def positional_encoding(x: torch.Tensor, num_freqs: int) -> torch.Tensor:
    """
    Sinusoidal positional encoding.

    Encodes input values using sinusoidal functions at multiple frequencies,
    following the NeRF positional encoding scheme.

    Args:
        x: (N,) or (N, D) input values to encode
        num_freqs: Number of frequency bands

    Returns:
        Encoded tensor with shape (N, D * (1 + 2 * num_freqs)) or (N, 1 + 2 * num_freqs)
        Format: [x, sin(pi*x), cos(pi*x), sin(2*pi*x), cos(2*pi*x), ...]
    """
    if x.dim() == 1:
        x = x.unsqueeze(-1)  # (N,) -> (N, 1)

    # Frequency bands: 2^0, 2^1, ..., 2^(num_freqs-1)
    freqs = 2.0 ** torch.arange(num_freqs, device=x.device, dtype=x.dtype)

    # x_freq shape: (N, D, num_freqs)
    x_freq = x.unsqueeze(-1) * freqs * math.pi

    # Compute sin and cos
    sin_enc = torch.sin(x_freq)  # (N, D, num_freqs)
    cos_enc = torch.cos(x_freq)  # (N, D, num_freqs)

    # Flatten and concatenate: [x, sin, cos]
    # Result shape: (N, D * (1 + 2 * num_freqs))
    encoded = torch.cat(
        [x, sin_enc.flatten(1), cos_enc.flatten(1)], dim=-1
    )

    return encoded


import re

def extract_animation_frame_numbers(mesh_paths):
    """
    Extract animation frame numbers from mesh_path strings.

    Args:
        mesh_paths: List of mesh path strings (e.g., ["mesh/frame_026.ply", "mesh/frame_067.ply"])

    Returns:
        List of animation frame numbers (e.g., [26, 67])
        Returns -1 for paths that don't match the pattern
    """
    anim_frames = []
    for mesh_path in mesh_paths:
        match = re.search(r'frame_(\d+)', mesh_path)
        if match:
            anim_frames.append(int(match.group(1)))
        else:
            anim_frames.append(-1)  # Static mesh or unknown format
    return anim_frames


def load_blender_train(data_dir, transforms_filename):
    with open(os.path.join(data_dir, transforms_filename), "r") as fp:
        meta = json.load(fp)

    cams = []
    image_paths = []
    mesh_paths = []
    for frame in meta["frames"]:
        cams.append(np.array(frame["transform_matrix"], dtype=np.float32))
        image_paths.append(os.path.join(data_dir, frame["file_path"] + ".png"))
        mesh_paths.append(frame.get("mesh_path", ""))

    if len(image_paths) == 0:
        raise RuntimeError("No frames found in transforms file")

    images = []
    for p in image_paths:
        with open(p, "rb") as imgin:
            image = np.array(Image.open(imgin), dtype=np.float32) / 255.0
        images.append(image[..., :3])

    images = np.stack(images, axis=0)
    h, w = images.shape[1], images.shape[2]

    camera_angle_x = float(meta["camera_angle_x"])
    focal = 0.5 * w / np.tan(0.5 * camera_angle_x)

    hwf = np.array([h, w, focal], dtype=np.float32)
    poses = np.stack(cams, axis=0)

    return {
        "images": images,
        "c2w": poses,
        "hwf": hwf,
        "image_paths": image_paths,
        "mesh_paths": mesh_paths,
    }


def pix2cam_matrix(height, width, focal, device):
    return torch.tensor(
        [
            [1.0 / focal, 0.0, -0.5 * width / focal],
            [0.0, -1.0 / focal, 0.5 * height / focal],
            [0.0, 0.0, -1.0],
        ],
        dtype=torch.float32,
        device=device,
    )


def generate_viewdirs(indices, ys, xs, c2w_all, pix2cam):
    pixel = torch.stack([xs.float(), ys.float()], dim=-1)
    ones = torch.ones(pixel.shape[0], 1, device=pixel.device)
    pixel_coords = torch.cat([pixel + 0.5, ones], dim=-1)

    cam_dirs = pixel_coords @ pix2cam.t()
    c2w = c2w_all[indices, :3, :3]
    ray_dirs = torch.einsum("bij,bj->bi", c2w, cam_dirs)
    ray_dirs = ray_dirs / torch.norm(ray_dirs, dim=-1, keepdim=True)
    return ray_dirs


class HybridTextureMLP(nn.Module):
    """
    Hybrid texture + MLP model for view-dependent rendering.

    When use_light_probe=True, the model samples from both a 2D texture and
    a 3D light probe grid, concatenating their features with view direction
    before the MLP. This allows learning spatially-varying lighting effects.

    When use_probe_delta=True, the model additionally uses a Conv3D network
    to generate per-frame deltas to the probe grid, enabling dynamic/temporal
    lighting effects (e.g., moving shadows for animated objects).

    Architecture (with probe, no delta):
        Input: uv (2), world_pos (3), viewdir (3)
        Texture: uv -> 8-dim features
        Probe: world_pos -> 16-dim features
        MLP: [8 + 16 + 3] = 27 -> 32 -> 32 -> 3 RGB

    Architecture (with probe + delta):
        Input: uv (2), world_pos (3), viewdir (3), frame_idx
        Texture: uv -> 8-dim features
        Probe Delta: Conv3D(probe + view_enc + frame_enc) -> delta grid
        Modified Probe: (base_probe + delta) -> 16-dim features
        MLP: [8 + 16 + 3 + 16] = 43 -> 64 -> 64 -> 3 RGB

    Architecture (without probe):
        Input: uv (2), viewdir (3)
        Texture: uv -> 8-dim features
        MLP: [8 + 3] = 11 -> 16 -> 16 -> 3 RGB
    """

    def __init__(
        self,
        texture_size,
        num_channels=8,
        hidden_features=(16, 16),
        use_light_probe=False,
        probe_resolution=16,
        probe_channels=16,
        probe_bounds_min=None,
        probe_bounds_max=None,
        use_probe_delta=False,
        probe_delta_period=200.0,
    ):
        super().__init__()
        self.use_light_probe = use_light_probe
        self.use_probe_delta = use_probe_delta
        self.probe_delta_period = probe_delta_period
        self.num_channels = num_channels
        self.probe_channels = probe_channels
        self.probe_resolution = probe_resolution

        self.texture = nn.Parameter(
            torch.zeros(1, num_channels, texture_size, texture_size)
        )

        # Light probe grid (optional)
        self.light_probe = None
        if use_light_probe:
            from model.light_probe import LightProbeGrid
            self.light_probe = LightProbeGrid(
                resolution=probe_resolution,
                num_channels=probe_channels,
                bounds_min=probe_bounds_min,
                bounds_max=probe_bounds_max,
            )

        # Dynamic probe delta network (optional)
        self.probe_delta_conv3d = None
        self.probe_delta_scale = None
        if use_probe_delta and use_light_probe:
            # Positional encoding dimensions
            # view_enc: viewdir(3) -> 3 * (1 + 2*2) = 15, truncated to 16
            # frame_enc: frame(1) -> 1 * (1 + 2*3) = 7, padded to 16
            self.view_enc_freqs = 2
            self.frame_enc_freqs = 3
            self.total_enc_dim = 32  # 16 (view) + 16 (frame)

            # Conv3D delta network for coarse probe modification
            hidden = max(32, probe_channels * 2)
            self.probe_delta_conv3d = nn.Sequential(
                nn.Conv3d(probe_channels + self.total_enc_dim, hidden, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv3d(hidden, hidden, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv3d(hidden, probe_channels, 3, padding=1),
                nn.Tanh(),
            )
            self.probe_delta_scale = nn.Parameter(torch.tensor(0.5))

        # Build MLP with appropriate input dimension
        if use_light_probe:
            if use_probe_delta:
                # texture(8) + probe(16) + viewdir(3) + frame_enc(16) = 43
                in_dim = num_channels + probe_channels + 3 + 16
                hidden_features = (64, 64)
            else:
                # texture(8) + probe(16) + viewdir(3) = 27
                in_dim = num_channels + probe_channels + 3
                hidden_features = (32, 32)
        else:
            if use_probe_delta:
                # texture(8) + viewdir(3) + frame_enc(16) = 27
                in_dim = num_channels + 3 + 16
                hidden_features = (32, 32)
            else:
                # texture(8) + viewdir(3) = 11
                in_dim = num_channels + 3

        layers = []
        last_dim = in_dim
        for feat in hidden_features:
            layers.append(nn.Linear(last_dim, feat))
            layers.append(nn.ReLU(inplace=True))
            last_dim = feat
        layers.append(nn.Linear(last_dim, 3))
        self.mlp = nn.Sequential(*layers)

    def forward(self, uv, viewdirs, world_pos=None, frame_idx=None):
        """
        Forward pass.

        Args:
            uv: (N, 2) UV coordinates in [-1, 1] range
            viewdirs: (N, 3) normalized view directions
            world_pos: (N, 3) world-space positions (required if use_light_probe=True)
            frame_idx: (N,) or scalar frame indices (used if use_probe_delta=True)

        Returns:
            (N, 3) RGB colors in [0, 1]
        """
        n = uv.shape[0]
        device = uv.device

        # Sample texture features
        grid = uv.view(1, n, 1, 2)
        tex_feat = F.grid_sample(
            self.texture,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        tex_feat = tex_feat.squeeze(0).squeeze(-1).permute(1, 0).contiguous()  # (N, 8)

        # Compute frame encoding (used for both probe delta AND MLP)
        frame_enc = None
        if self.use_probe_delta and frame_idx is not None:
            # Normalize frame index by period
            if isinstance(frame_idx, (int, float)):
                frame_idx = torch.full((n,), frame_idx, device=device, dtype=torch.float32)
            elif frame_idx.dim() == 0:
                frame_idx = frame_idx.expand(n).float()
            else:
                frame_idx = frame_idx.float()

            frame_norm = frame_idx / self.probe_delta_period  # (N,)

            # Positional encoding for frame
            frame_enc_raw = positional_encoding(frame_norm, self.frame_enc_freqs)  # (N, 7)
            # Pad or truncate to 16 dims
            if frame_enc_raw.shape[1] < 16:
                padding = torch.zeros(n, 16 - frame_enc_raw.shape[1], device=device)
                frame_enc = torch.cat([frame_enc_raw, padding], dim=-1)  # (N, 16)
            else:
                frame_enc = frame_enc_raw[:, :16]  # (N, 16)

        # Sample probe features
        probe_feat = None
        if self.use_light_probe and self.light_probe is not None:
            if world_pos is None:
                raise ValueError("world_pos required when use_light_probe=True")

            if self.use_probe_delta and frame_idx is not None and self.probe_delta_conv3d is not None:
                # Encode view direction for probe delta
                view_enc_raw = positional_encoding(viewdirs, self.view_enc_freqs)  # (N, 15)
                # Pad or truncate to 16 dims
                if view_enc_raw.shape[1] < 16:
                    padding = torch.zeros(n, 16 - view_enc_raw.shape[1], device=device)
                    view_enc = torch.cat([view_enc_raw, padding], dim=-1)  # (N, 16)
                else:
                    view_enc = view_enc_raw[:, :16]  # (N, 16)

                # Concatenate view and frame encodings
                total_enc = torch.cat([view_enc, frame_enc], dim=-1)  # (N, 32)

                # Broadcast encoding to 3D grid shape
                r = self.probe_resolution
                enc_3d = total_enc.view(n, self.total_enc_dim, 1, 1, 1).expand(
                    n, self.total_enc_dim, r, r, r
                )  # (N, 32, R, R, R)

                # Expand probe grid to batch
                probe_grid = self.light_probe.grid.expand(n, -1, -1, -1, -1)  # (N, C, R, R, R)

                # Concatenate probe + encoding
                conv_input = torch.cat([probe_grid, enc_3d], dim=1)  # (N, C+32, R, R, R)

                # Generate delta
                delta = self.probe_delta_conv3d(conv_input) * self.probe_delta_scale  # (N, C, R, R, R)

                # Sample with delta
                probe_feat = self.light_probe.sample_with_delta(world_pos, delta)  # (N, C)
            else:
                # Standard probe sampling (no delta)
                probe_feat = self.light_probe(world_pos)  # (N, probe_channels)

        # Build MLP input
        if self.use_light_probe and probe_feat is not None:
            if frame_enc is not None:
                # texture(8) + probe(16) + viewdir(3) + frame_enc(16) = 43
                x = torch.cat([tex_feat, probe_feat, viewdirs, frame_enc], dim=-1)
            else:
                # texture(8) + probe(16) + viewdir(3) = 27
                x = torch.cat([tex_feat, probe_feat, viewdirs], dim=-1)
        else:
            if frame_enc is not None:
                # texture(8) + viewdir(3) + frame_enc(16) = 27
                x = torch.cat([tex_feat, viewdirs, frame_enc], dim=-1)
            else:
                # texture(8) + viewdir(3) = 11
                x = torch.cat([tex_feat, viewdirs], dim=-1)

        rgb = torch.sigmoid(self.mlp(x))
        return rgb

    def compute_frame_delta(self, representative_viewdir, frame_idx):
        """
        Precompute probe delta for a single frame (for fast inference).

        Instead of computing delta for each pixel separately, we compute
        one shared delta using a representative viewdir (e.g., center pixel).
        This is ~100-200x faster for rendering.

        Args:
            representative_viewdir: (1, 3) or (3,) single view direction
            frame_idx: scalar or tensor frame index

        Returns:
            delta: (1, C, R, R, R) probe delta grid for this frame
            frame_enc: (1, 16) frame encoding for MLP input
        """
        if not self.use_probe_delta or self.probe_delta_conv3d is None:
            return None, None

        device = self.texture.device

        # Ensure viewdir is (1, 3)
        if representative_viewdir.dim() == 1:
            representative_viewdir = representative_viewdir.unsqueeze(0)

        # Ensure frame_idx is scalar tensor
        if isinstance(frame_idx, (int, float)):
            frame_idx_t = torch.tensor([frame_idx], device=device, dtype=torch.float32)
        elif frame_idx.dim() == 0:
            frame_idx_t = frame_idx.unsqueeze(0).float()
        else:
            frame_idx_t = frame_idx[:1].float()

        # Compute frame encoding
        frame_norm = frame_idx_t / self.probe_delta_period
        frame_enc_raw = positional_encoding(frame_norm, self.frame_enc_freqs)
        if frame_enc_raw.shape[1] < 16:
            padding = torch.zeros(1, 16 - frame_enc_raw.shape[1], device=device)
            frame_enc = torch.cat([frame_enc_raw, padding], dim=-1)
        else:
            frame_enc = frame_enc_raw[:, :16]

        # Compute view encoding
        view_enc_raw = positional_encoding(representative_viewdir, self.view_enc_freqs)
        if view_enc_raw.shape[1] < 16:
            padding = torch.zeros(1, 16 - view_enc_raw.shape[1], device=device)
            view_enc = torch.cat([view_enc_raw, padding], dim=-1)
        else:
            view_enc = view_enc_raw[:, :16]

        # Concatenate encodings
        total_enc = torch.cat([view_enc, frame_enc], dim=-1)  # (1, 32)

        # Broadcast to 3D grid
        r = self.probe_resolution
        enc_3d = total_enc.view(1, self.total_enc_dim, 1, 1, 1).expand(
            1, self.total_enc_dim, r, r, r
        )

        # Compute delta (single Conv3D call!)
        probe_grid = self.light_probe.grid  # (1, C, R, R, R)
        conv_input = torch.cat([probe_grid, enc_3d], dim=1)
        delta = self.probe_delta_conv3d(conv_input) * self.probe_delta_scale

        return delta, frame_enc

    def forward_with_precomputed_delta(self, uv, viewdirs, world_pos, precomputed_delta, precomputed_frame_enc):
        """
        Fast forward pass using precomputed probe delta.

        This skips the expensive per-pixel Conv3D computation by using
        a shared delta for all pixels in a frame.

        Args:
            uv: (N, 2) UV coordinates in [-1, 1]
            viewdirs: (N, 3) view directions
            world_pos: (N, 3) world positions
            precomputed_delta: (1, C, R, R, R) from compute_frame_delta
            precomputed_frame_enc: (1, 16) from compute_frame_delta

        Returns:
            (N, 3) RGB colors
        """
        n = uv.shape[0]
        device = uv.device

        # Sample texture (same as forward)
        grid = uv.view(1, n, 1, 2)
        tex_feat = F.grid_sample(
            self.texture, grid, mode="bilinear",
            padding_mode="border", align_corners=True
        )
        tex_feat = tex_feat.squeeze(0).squeeze(-1).permute(1, 0).contiguous()

        # Sample probe with precomputed delta
        if self.use_light_probe and self.light_probe is not None and precomputed_delta is not None:
            # Expand delta to batch size for sampling
            # Note: sample_with_delta expects (N, C, R, R, R) but we have (1, C, R, R, R)
            # We use a modified sampling that broadcasts the single delta
            probe_feat = self.light_probe.sample_with_shared_delta(world_pos, precomputed_delta)
        elif self.use_light_probe and self.light_probe is not None:
            probe_feat = self.light_probe(world_pos)
        else:
            probe_feat = None

        # Expand frame_enc to batch size
        frame_enc = precomputed_frame_enc.expand(n, -1) if precomputed_frame_enc is not None else None

        # Build MLP input
        if self.use_light_probe and probe_feat is not None:
            if frame_enc is not None:
                x = torch.cat([tex_feat, probe_feat, viewdirs, frame_enc], dim=-1)
            else:
                x = torch.cat([tex_feat, probe_feat, viewdirs], dim=-1)
        else:
            if frame_enc is not None:
                x = torch.cat([tex_feat, viewdirs, frame_enc], dim=-1)
            else:
                x = torch.cat([tex_feat, viewdirs], dim=-1)

        rgb = torch.sigmoid(self.mlp(x))
        return rgb


def export_mobilenerf_assets(model, texture_size, export_dir, mesh_path=None):
    """
    Export assets in MobileNeRF Unity Viewer compatible format.

    Expected by Unity viewer (julienkay/MobileNeRF-Unity-Viewer):
    - shape0.obj (mesh)
    - shape0.pngfeat0.png, shape0.pngfeat1.png (feature textures)
    - mlp.json (MLP weights)
    - probe.json (light probe data, if enabled)
    """
    os.makedirs(export_dir, exist_ok=True)

    # Export feature textures
    with torch.no_grad():
        tex = model.texture.detach().cpu()[0]
    tex = tex.clamp(0.0, 1.0)
    tex_uint8 = (tex * 255.0).round().to(dtype=torch.uint8)
    tex_uint8 = tex_uint8.permute(1, 2, 0).numpy()

    h, w, c = tex_uint8.shape
    if c % 4 != 0:
        raise RuntimeError("num_channels must be divisible by 4 for feat0/feat1 export")

    out_feat_num = c // 4
    for i in range(out_feat_num):
        ff = np.zeros((h, w, 4), dtype=np.uint8)
        # BGR swap for Unity compatibility
        ff[..., 0] = tex_uint8[..., i * 4 + 2]
        ff[..., 1] = tex_uint8[..., i * 4 + 1]
        ff[..., 2] = tex_uint8[..., i * 4 + 0]
        ff[..., 3] = tex_uint8[..., i * 4 + 3]
        img = Image.fromarray(ff, mode="RGBA")
        # Unity viewer expects: shape0.pngfeat0.png, shape0.pngfeat1.png
        img.save(os.path.join(export_dir, f"shape0.pngfeat{i}.png"))

    # Export MLP weights
    mlp = model.mlp
    w0 = mlp[0].weight.detach().cpu().t().numpy().tolist()
    b0 = mlp[0].bias.detach().cpu().numpy().tolist()
    w1 = mlp[2].weight.detach().cpu().t().numpy().tolist()
    b1 = mlp[2].bias.detach().cpu().numpy().tolist()
    w2 = mlp[4].weight.detach().cpu().t().numpy().tolist()
    b2 = mlp[4].bias.detach().cpu().numpy().tolist()

    mlp_params = {
        "0_weights": w0,
        "1_weights": w1,
        "2_weights": w2,
        "0_bias": b0,
        "1_bias": b1,
        "2_bias": b2,
        "obj_num": 1,
        "use_light_probe": model.use_light_probe,
    }

    scene_params_path = os.path.join(export_dir, "mlp.json")
    with open(scene_params_path, "w", encoding="utf-8") as f:
        json.dump(mlp_params, f)

    # Export light probe data if enabled
    if model.use_light_probe and model.light_probe is not None:
        probe_data = model.light_probe.export_to_dict()
        probe_path = os.path.join(export_dir, "probe.json")
        with open(probe_path, "w", encoding="utf-8") as f:
            json.dump(probe_data, f)
        print(f"   Exported light probe to: {probe_path}")

    # Copy mesh file as shape0.obj (Unity viewer naming convention)
    if mesh_path and os.path.exists(mesh_path):
        import shutil
        dst_mesh = os.path.join(export_dir, "shape0.obj")
        shutil.copy2(mesh_path, dst_mesh)
        print(f"   Copied mesh to: {dst_mesh}")

    print(f"   Exported Unity-compatible assets to: {export_dir}")


def sample_batch(
    batch_size,
    uv_all,
    mask_all,
    images,
    c2w_all,
    pix2cam,
    device,
    pos_all=None,
    return_frame_idx=False,
    view_to_anim=None,
):
    """
    Sample a random batch of pixels for training.

    Args:
        batch_size: Number of pixels to sample
        uv_all: (N_views, H, W, 2) UV coordinates
        mask_all: (N_views, H, W) valid pixel mask
        images: (N_views, H, W, 3) ground truth RGB
        c2w_all: (N_views, 4, 4) camera-to-world matrices
        pix2cam: (3, 3) pixel-to-camera matrix
        device: torch device
        pos_all: (N_views, H, W, 3) world positions (optional)
        return_frame_idx: If True, also return frame indices
        view_to_anim: (N_views,) tensor mapping view indices to animation frame numbers
                      If provided, returns actual animation frames instead of view indices

    Returns:
        Tuple of (uv_grid, viewdirs, gt_rgb) or (uv_grid, viewdirs, gt_rgb, world_pos)
        or (uv_grid, viewdirs, gt_rgb, world_pos, frame_idx) if return_frame_idx=True
        Returns None if no valid pixels sampled
    """
    num_views, height, width, _ = uv_all.shape

    view_inds = torch.randint(0, num_views, (batch_size,), device=device)
    y_inds = torch.randint(0, height, (batch_size,), device=device)
    x_inds = torch.randint(0, width, (batch_size,), device=device)

    valid = mask_all[view_inds, y_inds, x_inds]
    if valid.sum() == 0:
        return None

    view_inds = view_inds[valid]
    y_inds = y_inds[valid]
    x_inds = x_inds[valid]

    uv = uv_all[view_inds, y_inds, x_inds]
    uv_grid = uv * 2.0 - 1.0

    gt_rgb = images[view_inds, y_inds, x_inds]

    viewdirs = generate_viewdirs(view_inds, y_inds, x_inds, c2w_all, pix2cam)

    # Convert view indices to animation frame numbers if mapping provided
    frame_idx = view_inds
    if view_to_anim is not None:
        frame_idx = view_to_anim[view_inds]

    if pos_all is not None:
        world_pos = pos_all[view_inds, y_inds, x_inds]
        if return_frame_idx:
            return uv_grid, viewdirs, gt_rgb, world_pos, frame_idx
        return uv_grid, viewdirs, gt_rgb, world_pos

    if return_frame_idx:
        return uv_grid, viewdirs, gt_rgb, frame_idx

    return uv_grid, viewdirs, gt_rgb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_root",
        type=str,
        default=os.path.join("data", "custom", "MyNeRFData"),
    )
    parser.add_argument(
        "--uv_path",
        type=str,
        default=os.path.join("data", "custom", "MyNeRFData", "uv_lookup.npz"),
    )
    parser.add_argument(
        "--transforms",
        type=str,
        default="transforms_train.json",
    )
    parser.add_argument(
        "--texture_size",
        type=int,
        default=2048,
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2048,
    )
    parser.add_argument(
        "--num_iters",
        type=int,
        default=150000,
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=os.path.join("weights", "hybrid_texture_mlp.pth"),
    )
    parser.add_argument(
        "--export_dir",
        type=str,
        default=os.path.join("weights", "mobilenerf_export"),
    )
    parser.add_argument(
        "--mesh_path",
        type=str,
        default=os.path.join("data", "custom", "MyNeRFData", "sponza_gt_unwarpped.obj"),
        help="Path to unwrapped OBJ mesh (will be copied to export_dir as shape0.obj)",
    )
    parser.add_argument(
        "--downscale",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
    )
    parser.add_argument(
        "--use_light_probe",
        action="store_true",
        help="Enable light probe grid for spatially-varying lighting features",
    )
    parser.add_argument(
        "--probe_resolution",
        type=int,
        default=16,
        help="Light probe grid resolution (default 16 -> 16^3 voxels)",
    )
    parser.add_argument(
        "--probe_channels",
        type=int,
        default=16,
        help="Number of feature channels in light probe (default 16)",
    )
    parser.add_argument(
        "--use_probe_delta",
        action="store_true",
        help="Enable dynamic probe delta network for temporal/animated lighting",
    )
    parser.add_argument(
        "--probe_delta_period",
        type=float,
        default=None,
        help="Temporal period for probe delta encoding (auto-calculated from frame count if not specified)",
    )
    parser.add_argument(
        "--sample_view_idx",
        type=int,
        default=0,
        help="Which view index (r_X) to use for sample rendering during training",
    )
    args = parser.parse_args()

    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA 不可用，但 device=cuda 被指定")
        device = torch.device("cuda")
    elif args.device == "cpu":
        device = torch.device("cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("Hybrid training device:", device)
    print("torch.cuda.is_available:", torch.cuda.is_available())
    if device.type == "cuda":
        print("CUDA version:", torch.version.cuda)
        print("GPU name:", torch.cuda.get_device_name(0))

    uv_data = np.load(args.uv_path, allow_pickle=True, mmap_mode="r")
    uv_np = uv_data["uv"]
    mask_np = uv_data["mask"]

    # Load position data for light probe (if available and enabled)
    pos_all = None
    probe_bounds_min = None
    probe_bounds_max = None
    if args.use_light_probe:
        if "pos" not in uv_data:
            raise RuntimeError(
                "Light probe enabled but 'pos' not found in uv_lookup.npz. "
                "Re-run 01_preprocess_raster.py to generate position maps."
            )
        pos_np = uv_data["pos"]
        probe_bounds_min = torch.from_numpy(uv_data["probe_bounds_min"])
        probe_bounds_max = torch.from_numpy(uv_data["probe_bounds_max"])
        print(f"Light probe enabled: bounds_min={probe_bounds_min.tolist()}, bounds_max={probe_bounds_max.tolist()}")

    ds = max(1, int(args.downscale))
    if ds > 1:
        uv_np = uv_np[:, ::ds, ::ds, :]
        mask_np = mask_np[:, ::ds, ::ds]
        if args.use_light_probe:
            pos_np = pos_np[:, ::ds, ::ds, :]

    uv_all = torch.from_numpy(uv_np).to(device=device, dtype=torch.float32)
    mask_all = torch.from_numpy(mask_np).to(device=device)
    if args.use_light_probe:
        pos_all = torch.from_numpy(pos_np).to(device=device, dtype=torch.float32)

    train_data = load_blender_train(args.data_root, args.transforms)
    images_np = train_data["images"]
    c2w_np = train_data["c2w"]
    hwf = train_data["hwf"]
    mesh_paths = train_data["mesh_paths"]

    # -------------------------------------------------------------------------
    # Build view_idx → animation_frame mapping for probe delta
    # This ensures we use actual animation frame numbers, not shuffled view indices
    # -------------------------------------------------------------------------
    view_to_anim = None
    max_anim_frame = 0
    if args.use_probe_delta and mesh_paths:
        anim_frames = extract_animation_frame_numbers(mesh_paths)
        valid_frames = [f for f in anim_frames if f >= 0]
        if valid_frames:
            max_anim_frame = max(valid_frames)
            view_to_anim = torch.tensor(anim_frames, dtype=torch.float32, device=device)
            print(f"📊 Animation frame mapping:")
            print(f"   - {len(valid_frames)} views with valid animation frames")
            print(f"   - Animation frame range: 0 to {max_anim_frame}")
            print(f"   - First 5 mappings: {list(zip(range(5), anim_frames[:5]))}")
        else:
            print("⚠️ No valid animation frames found in mesh_paths. Using view indices.")

    if ds > 1:
        images_np = images_np[:, ::ds, ::ds, :]
        hwf[0] = hwf[0] / ds
        hwf[1] = hwf[1] / ds
        hwf[2] = hwf[2] / ds

    images = torch.from_numpy(images_np).to(device=device, dtype=torch.float32)
    c2w_all = torch.from_numpy(c2w_np).to(device=device, dtype=torch.float32)

    h, w, focal = hwf
    pix2cam = pix2cam_matrix(h, w, focal, device)

    # -------------------------------------------------------------------------
    # Calculate probe_delta_period based on actual animation frame range
    # Use max_anim_frame + 1 (actual frame range), not num_views (shuffled count)
    # -------------------------------------------------------------------------
    num_frames = uv_all.shape[0]
    probe_delta_period = args.probe_delta_period
    if args.use_probe_delta and probe_delta_period is None:
        if max_anim_frame > 0:
            probe_delta_period = float(max_anim_frame + 1)
            print(f"✅ probe_delta_period = {probe_delta_period} (from max animation frame {max_anim_frame})")
        else:
            probe_delta_period = float(num_frames)
            print(f"⚠️ probe_delta_period = {probe_delta_period} (fallback to view count)")
    elif args.use_probe_delta:
        print(f"Using specified probe_delta_period = {probe_delta_period}")

    model = HybridTextureMLP(
        texture_size=args.texture_size,
        use_light_probe=args.use_light_probe,
        probe_resolution=args.probe_resolution,
        probe_channels=args.probe_channels,
        probe_bounds_min=probe_bounds_min,
        probe_bounds_max=probe_bounds_max,
        use_probe_delta=args.use_probe_delta,
        probe_delta_period=probe_delta_period if probe_delta_period else 200.0,
    ).to(device)

    if args.use_light_probe:
        total_params = sum(p.numel() for p in model.parameters())
        mlp_params = sum(p.numel() for p in model.mlp.parameters())
        probe_params = sum(p.numel() for p in model.light_probe.parameters())
        if args.use_probe_delta and model.probe_delta_conv3d is not None:
            delta_params = sum(p.numel() for p in model.probe_delta_conv3d.parameters())
            delta_params += 1  # probe_delta_scale
            print(f"Model parameters: total={total_params}, MLP={mlp_params}, probe={probe_params}, delta_net={delta_params}")
        else:
            print(f"Model parameters: total={total_params}, MLP={mlp_params}, probe={probe_params}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    start_step = 0
    
    # -------------------------------------------------------------------------
    # Resume from checkpoint if available
    # -------------------------------------------------------------------------
    if os.path.exists(args.checkpoint):
        print(f"🔄 Found checkpoint: {args.checkpoint}. Resuming training...")
        try:
            checkpoint_data = torch.load(args.checkpoint, map_location=device, weights_only=False)
            model.load_state_dict(checkpoint_data["model_state_dict"])
            optimizer.load_state_dict(checkpoint_data["optimizer_state_dict"])
            if "step" in checkpoint_data:
                start_step = checkpoint_data["step"]
            print(f"✅ Resumed from step {start_step}")
        except Exception as e:
            print(f"⚠️ Failed to load checkpoint: {e}. Starting from scratch.")
    else:
        print("🆕 No checkpoint found. Starting fresh training.")

    os.makedirs(os.path.dirname(args.checkpoint), exist_ok=True)

    # -------------------------------------------------------------------------
    # Training Loop
    # -------------------------------------------------------------------------
    if start_step >= args.num_iters:
        print(f"🎉 Training already completed ({start_step}/{args.num_iters} steps).")
        return

    pbar = tqdm(range(start_step, args.num_iters), desc="Hybrid training", unit="iter", initial=start_step, total=args.num_iters)
    for step in pbar:
        batch = sample_batch(
            args.batch_size,
            uv_all,
            mask_all,
            images,
            c2w_all,
            pix2cam,
            device,
            pos_all=pos_all,
            return_frame_idx=args.use_probe_delta,
            view_to_anim=view_to_anim,
        )
        if batch is None:
            continue

        # Unpack batch based on available data
        frame_idx = None
        if pos_all is not None:
            if args.use_probe_delta:
                uv_grid, viewdirs, gt_rgb, world_pos, frame_idx = batch
            else:
                uv_grid, viewdirs, gt_rgb, world_pos = batch
        else:
            if args.use_probe_delta:
                uv_grid, viewdirs, gt_rgb, frame_idx = batch
            else:
                uv_grid, viewdirs, gt_rgb = batch
            world_pos = None

        optimizer.zero_grad()
        pred_rgb = model(uv_grid, viewdirs, world_pos=world_pos, frame_idx=frame_idx)
        loss = criterion(pred_rgb, gt_rgb)
        loss.backward()
        optimizer.step()

        # ========== OPTIMIZED: Better LR decay schedule ==========
        # Changed from aggressive decay (0.01^t) to gentler decay (0.1^t)
        # This keeps learning rate higher in later stages for better convergence
        t = (step + 1) / float(args.num_iters)
        lr_now = args.lr * (0.1 ** t)  # Changed: 0.01 → 0.1 (10x higher in late stage)

        # Optional: Warmup from checkpoint resume
        if start_step > 0 and step < start_step + 5000:
            # Gradually increase LR over first 5k steps after resume
            warmup_progress = (step - start_step) / 5000.0
            base_lr_at_resume = args.lr * (0.1 ** (start_step / float(args.num_iters)))
            lr_now = base_lr_at_resume + (lr_now - base_lr_at_resume) * warmup_progress

        for g in optimizer.param_groups:
            g["lr"] = lr_now

        if (step + 1) % 100 == 0:
            with torch.no_grad():
                mse = loss.item()
                psnr = -10.0 * np.log10(mse + 1e-8)
            pbar.set_postfix(loss=mse, psnr=psnr, lr=lr_now)

        if (step + 1) % 10000 == 0:
            with torch.no_grad():
                mse = loss.item()
                psnr = -10.0 * np.log10(mse + 1e-8)
            print(
                "[Hybrid] step {}/{} | loss={:.6f} | psnr={:.2f} | lr={:.2e}".format(
                    step + 1, args.num_iters, mse, psnr, lr_now
                )
            )

            checkpoint_dict = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "step": step + 1,
                "hwf": hwf,
            }
            # Save animation frame mapping for inference
            if view_to_anim is not None:
                checkpoint_dict["view_to_anim"] = view_to_anim.cpu()
                checkpoint_dict["probe_delta_period"] = probe_delta_period
                checkpoint_dict["max_anim_frame"] = max_anim_frame
            torch.save(checkpoint_dict, args.checkpoint)

            export_mobilenerf_assets(model, args.texture_size, args.export_dir, args.mesh_path)

            view_idx = args.sample_view_idx
            model.eval()
            with torch.no_grad():
                uv_view = uv_all[view_idx]
                h_img, w_img, _ = uv_view.shape
                uv_grid_full = uv_view * 2.0 - 1.0
                uv_flat = uv_grid_full.view(-1, 2)

                # Get world positions for this view if using light probe
                pos_flat = None
                if pos_all is not None:
                    pos_view = pos_all[view_idx]
                    pos_flat = pos_view.view(-1, 3)

                ys, xs = torch.meshgrid(
                    torch.arange(h_img, device=device),
                    torch.arange(w_img, device=device),
                    indexing="ij",
                )
                ys_flat = ys.reshape(-1)
                xs_flat = xs.reshape(-1)
                indices = torch.full_like(ys_flat, view_idx, dtype=torch.long)
                viewdirs_full = generate_viewdirs(indices, ys_flat, xs_flat, c2w_all, pix2cam)

                preds = []
                chunk = 8192
                for i in range(0, uv_flat.shape[0], chunk):
                    u = uv_flat[i:i + chunk]
                    v = viewdirs_full[i:i + chunk]
                    p = pos_flat[i:i + chunk] if pos_flat is not None else None
                    # Pass actual animation frame number (not view_idx)
                    f_idx = None
                    if args.use_probe_delta:
                        if view_to_anim is not None:
                            f_idx = view_to_anim[view_idx].item()
                        else:
                            f_idx = view_idx
                    preds.append(model(u, v, world_pos=p, frame_idx=f_idx))
                preds = torch.cat(preds, dim=0)
                preds = preds.view(h_img, w_img, 3).clamp(0.0, 1.0).cpu().numpy()

                gt = images[view_idx].cpu().numpy()

                pred_img = (preds * 255.0).astype(np.uint8)
                gt_img = (gt * 255.0).astype(np.uint8)

                samples_dir = "samples"
                if not os.path.exists(samples_dir):
                    os.makedirs(samples_dir)

                Image.fromarray(pred_img).save(
                    os.path.join(samples_dir, "hybrid_step_{:06d}_view_{:03d}_pred.png".format(step + 1, view_idx))
                )
                Image.fromarray(gt_img).save(
                    os.path.join(samples_dir, "hybrid_step_{:06d}_view_{:03d}_gt.png".format(step + 1, view_idx))
                )
            model.train()


if __name__ == "__main__":
    main()
