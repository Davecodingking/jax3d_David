"""
Pre-render UV maps for animated datasets.

Uses fast_uv_renderer (Numba JIT) to rasterize per-frame PLY meshes
and outputs our uv_lookup.npz format for 02_train_hybrid.py.

Supports animated datasets where each image frame has a corresponding mesh,
specified via 'mesh_path' field in transforms JSON.

Usage:
    python prerender_animated.py --data data/hotdog_animated
    python prerender_animated.py --data data/hotdog_animated --scale 0.5 --split train
"""

import os
import json
import numpy as np
from PIL import Image
from tqdm import tqdm

from ply_loader import load_ply
from fast_uv_renderer import FastUVRenderer


def prerender_animated_to_npz(data_dir, output_file, scale=1.0, split='train'):
    """
    Pre-render animated dataset to single uv_lookup.npz file.

    Args:
        data_dir: Path to dataset (e.g., data/hotdog_animated)
        output_file: Output path (e.g., data/hotdog_animated/uv_lookup.npz)
        scale: Image scale factor (1.0 = full resolution)
        split: Which split to process ('train', 'val', 'test')
    """
    # Load transforms
    transforms_file = os.path.join(data_dir, f'transforms_{split}.json')
    if not os.path.exists(transforms_file):
        raise FileNotFoundError(f"Transforms file not found: {transforms_file}")

    with open(transforms_file, 'r') as f:
        transforms = json.load(f)

    camera_angle_x = transforms['camera_angle_x']
    frames = transforms['frames']

    if len(frames) == 0:
        raise ValueError("No frames found in transforms file")

    # Get image dimensions from first frame
    first_frame = frames[0]
    first_img_path = os.path.join(data_dir, first_frame['file_path'] + '.png')
    if not os.path.exists(first_img_path):
        # Try without .png extension (some datasets include it)
        first_img_path = os.path.join(data_dir, first_frame['file_path'])

    first_img = Image.open(first_img_path)
    orig_w, orig_h = first_img.size
    width = int(orig_w * scale)
    height = int(orig_h * scale)

    print(f"Dataset: {data_dir}")
    print(f"Processing {len(frames)} frames at {width}x{height} (scale={scale})")

    # Mesh cache - avoid reloading same mesh
    mesh_cache = {}

    # Renderer cache - avoid recreating for same mesh
    renderer_cache = {}

    # Collect all data
    all_uv = []
    all_pos = []
    all_mask = []
    image_paths = []

    # For probe bounds, collect bounds from all unique meshes
    all_bounds_min = []
    all_bounds_max = []

    for i, frame in enumerate(tqdm(frames, desc="Rasterizing")):
        # Get mesh for this frame
        mesh_path = frame.get('mesh_path')
        if mesh_path:
            # Animated: per-frame mesh specified in JSON
            full_mesh_path = os.path.join(data_dir, mesh_path + '.ply')
        else:
            # Static: single mesh (fallback)
            full_mesh_path = os.path.join(data_dir, 'mesh_uv.ply')

        if not os.path.exists(full_mesh_path):
            raise FileNotFoundError(f"Mesh not found: {full_mesh_path}")

        # Load mesh (cached)
        if full_mesh_path not in mesh_cache:
            mesh = load_ply(full_mesh_path)
            mesh_cache[full_mesh_path] = mesh
            bmin, bmax = mesh.bounds
            all_bounds_min.append(bmin)
            all_bounds_max.append(bmax)
            print(f"  Loaded mesh: {os.path.basename(full_mesh_path)} "
                  f"({mesh.num_vertices} verts, {mesh.num_faces} faces)")

        mesh = mesh_cache[full_mesh_path]

        # Get or create renderer for this mesh
        if full_mesh_path not in renderer_cache:
            renderer_cache[full_mesh_path] = FastUVRenderer(mesh, width, height)

        renderer = renderer_cache[full_mesh_path]

        # Render UV and position maps
        transform_matrix = np.array(frame['transform_matrix'], dtype=np.float32)
        uv_map, mask, pos_map = renderer.render_uv_map_from_nerf_transform(
            transform_matrix, camera_angle_x
        )

        # Store
        all_uv.append(uv_map)
        all_pos.append(pos_map)
        all_mask.append(mask)

        # Image path (for training dataloader)
        file_path = frame['file_path']
        if not file_path.endswith('.png'):
            file_path = file_path + '.png'
        img_path = os.path.join(data_dir, file_path)
        image_paths.append(img_path)

    # Stack arrays: (N_views, H, W, C)
    uv_all = np.stack(all_uv, axis=0)
    pos_all = np.stack(all_pos, axis=0)
    mask_all = np.stack(all_mask, axis=0)

    # Compute probe bounds from all meshes with padding
    probe_bounds_min = np.minimum.reduce(all_bounds_min) - 0.1
    probe_bounds_max = np.maximum.reduce(all_bounds_max) + 0.1

    # Compute focal length from FOV
    focal = 0.5 * width / np.tan(0.5 * camera_angle_x)

    # Save in our format (compatible with 02_train_hybrid.py)
    np.savez_compressed(
        output_file,
        uv=uv_all.astype(np.float32),
        pos=pos_all.astype(np.float32),
        mask=mask_all.astype(np.bool_),
        probe_bounds_min=probe_bounds_min.astype(np.float32),
        probe_bounds_max=probe_bounds_max.astype(np.float32),
        image_paths=np.array(image_paths),
        H=np.int32(height),
        W=np.int32(width),
        focal=np.float32(focal),
    )

    print(f"\nSaved to: {output_file}")
    print(f"  uv shape: {uv_all.shape}")
    print(f"  pos shape: {pos_all.shape}")
    print(f"  mask coverage: {mask_all.sum() / mask_all.size * 100:.1f}%")
    print(f"  probe_bounds: [{probe_bounds_min}] to [{probe_bounds_max}]")
    print(f"  focal: {focal:.2f}")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(
        description="Pre-render UV maps for animated datasets"
    )
    parser.add_argument(
        '--data', type=str, required=True,
        help="Path to dataset directory (e.g., data/hotdog_animated)"
    )
    parser.add_argument(
        '--output', type=str, default=None,
        help="Output path for uv_lookup.npz (default: <data>/uv_lookup.npz)"
    )
    parser.add_argument(
        '--scale', type=float, default=1.0,
        help="Image scale factor (default: 1.0 = full resolution)"
    )
    parser.add_argument(
        '--split', type=str, default='train',
        help="Which split to process (default: train)"
    )
    args = parser.parse_args()

    if args.output is None:
        args.output = os.path.join(args.data, 'uv_lookup.npz')

    prerender_animated_to_npz(args.data, args.output, args.scale, args.split)
