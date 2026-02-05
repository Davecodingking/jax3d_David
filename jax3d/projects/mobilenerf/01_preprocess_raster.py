import os
import json
import argparse

import numpy as np
from PIL import Image

import torch
from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes
from pytorch3d.renderer import PerspectiveCameras, RasterizationSettings, MeshRasterizer
from tqdm import tqdm


def load_blender_cameras(data_dir, transforms_filename):
    with open(os.path.join(data_dir, transforms_filename), "r") as fp:
        meta = json.load(fp)

    cams = []
    image_paths = []
    for frame in meta["frames"]:
        cams.append(np.array(frame["transform_matrix"], dtype=np.float32))
        image_paths.append(os.path.join(data_dir, frame["file_path"] + ".png"))

    if len(image_paths) == 0:
        raise RuntimeError("No frames found in transforms file")

    with open(image_paths[0], "rb") as imgin:
        image = np.array(Image.open(imgin), dtype=np.float32) / 255.0

    h, w = image.shape[0], image.shape[1]
    camera_angle_x = float(meta["camera_angle_x"])
    focal = 0.5 * w / np.tan(0.5 * camera_angle_x)

    hwf = np.array([h, w, focal], dtype=np.float32)
    poses = np.stack(cams, axis=0)

    return {
        "images_hwf": hwf,
        "c2w": poses,
        "image_paths": image_paths,
    }


def build_cameras(c2w, hwf, device, legacy_xy_flip):
    h, w, focal = hwf
    h = float(h)
    w = float(w)
    focal = float(focal)

    c2w_torch = torch.from_numpy(c2w).to(device=device, dtype=torch.float32)
    if c2w_torch.shape[-2:] == (3, 4):
        bottom = torch.tensor(
            [[[0.0, 0.0, 0.0, 1.0]]],
            dtype=c2w_torch.dtype,
            device=device,
        )
        bottom = bottom.expand(c2w_torch.shape[0], -1, -1)
        c2w_torch = torch.cat([c2w_torch, bottom], dim=-2)

    R_c2w = c2w_torch[:, :3, :3]
    t_c2w = c2w_torch[:, :3, 3]

    R_w2c = R_c2w.transpose(1, 2)
    t_w2c = -torch.bmm(R_w2c, t_c2w.unsqueeze(-1)).squeeze(-1)

    if legacy_xy_flip:
        R_w2c[:, 0, :] *= -1.0
        R_w2c[:, 1, :] *= -1.0
        t_w2c[:, 0] *= -1.0
        t_w2c[:, 1] *= -1.0

    R = R_w2c.transpose(1, 2)
    T = t_w2c

    fx = focal
    fy = focal
    cx = 0.5 * w
    cy = 0.5 * h

    focal_length = torch.tensor(
        [[fx, fy]], dtype=torch.float32, device=device
    ).expand(c2w_torch.shape[0], -1)
    principal_point = torch.tensor(
        [[cx, cy]], dtype=torch.float32, device=device
    ).expand(c2w_torch.shape[0], -1)
    image_size = torch.tensor(
        [[h, w]], dtype=torch.float32, device=device
    ).expand(c2w_torch.shape[0], -1)

    cameras = PerspectiveCameras(
        R=R,
        T=T,
        focal_length=focal_length,
        principal_point=principal_point,
        image_size=image_size,
        in_ndc=False,
        device=device,
    )

    return cameras


def compute_uv_lookup(mesh, cameras, hwf, device, verts_uvs, faces_uvs, verts, faces_verts, faces_per_pixel=1):
    """
    Compute UV and position lookup maps for all camera views.

    Args:
        mesh: PyTorch3D Meshes object
        cameras: PerspectiveCameras for all views
        hwf: (height, width, focal) array
        device: torch device
        verts_uvs: UV coordinates per vertex
        faces_uvs: UV indices per face
        verts: Vertex positions (N_verts, 3)
        faces_verts: Vertex indices per face (N_faces, 3)
        faces_per_pixel: Number of faces to track per pixel

    Returns:
        all_uv: (N_views, H, W, 2) UV coordinates
        all_mask: (N_views, H, W) valid pixel mask
        all_pos: (N_views, H, W, 3) world-space positions
    """
    h, w, _ = hwf.astype(np.int32)
    raster_settings = RasterizationSettings(
        image_size=(int(h), int(w)),
        blur_radius=0.0,
        faces_per_pixel=faces_per_pixel,
        cull_backfaces=False,
        bin_size=None,
    )

    rasterizer = MeshRasterizer(raster_settings=raster_settings)

    all_uv = []
    all_mask = []
    all_pos = []

    num_views = cameras.R.shape[0]
    for i in tqdm(range(num_views), desc="Rasterizing views", unit="view"):
        cam_idx = torch.tensor([i], device=device, dtype=torch.long)
        cam = cameras[cam_idx]
        fragments = rasterizer(meshes_world=mesh, cameras=cam)
        pix_to_face = fragments.pix_to_face[0, ..., 0]
        bary_coords = fragments.bary_coords[0, ..., 0, :]

        pix_to_face_flat = pix_to_face.view(-1)
        bary_flat = bary_coords.view(-1, 3)

        valid = pix_to_face_flat >= 0

        if not torch.any(valid):
            uv = torch.zeros(
                pix_to_face_flat.shape[0],
                2,
                dtype=torch.float32,
                device=device,
            )
            pos = torch.zeros(
                pix_to_face_flat.shape[0],
                3,
                dtype=torch.float32,
                device=device,
            )
            mask = valid.view(int(h), int(w))
            uv = uv.view(int(h), int(w), 2)
            pos = pos.view(int(h), int(w), 3)
        else:
            face_idx_valid = pix_to_face_flat[valid]
            face_uv_idx = faces_uvs[face_idx_valid]
            verts_uv = verts_uvs[face_uv_idx]

            bary_valid = bary_flat[valid].unsqueeze(-1)
            uv_valid = (verts_uv * bary_valid).sum(dim=1)

            # Position interpolation using barycentric coordinates
            face_vert_idx = faces_verts[face_idx_valid]  # (N_valid, 3)
            verts_pos = verts[face_vert_idx]  # (N_valid, 3, 3) - XYZ of 3 vertices
            pos_valid = (verts_pos * bary_valid).sum(dim=1)  # (N_valid, 3)

            uv = torch.zeros(
                pix_to_face_flat.shape[0],
                2,
                dtype=torch.float32,
                device=device,
            )
            uv[valid] = uv_valid

            pos = torch.zeros(
                pix_to_face_flat.shape[0],
                3,
                dtype=torch.float32,
                device=device,
            )
            pos[valid] = pos_valid

            uv = uv.view(int(h), int(w), 2)
            pos = pos.view(int(h), int(w), 3)
            mask = valid.view(int(h), int(w))

        all_uv.append(uv.cpu().numpy())
        all_mask.append(mask.cpu().numpy())
        all_pos.append(pos.cpu().numpy())

    all_uv = np.stack(all_uv, axis=0)
    all_mask = np.stack(all_mask, axis=0)
    all_pos = np.stack(all_pos, axis=0)

    return all_uv, all_mask, all_pos


def compute_probe_bounds(verts, padding=0.1):
    """
    Compute probe volume bounds from mesh vertices.

    Args:
        verts: (N, 3) vertex positions
        padding: Fractional padding to add around the mesh (default 10%)

    Returns:
        bounds_min: (3,) minimum corner
        bounds_max: (3,) maximum corner
    """
    if isinstance(verts, torch.Tensor):
        verts_np = verts.cpu().numpy()
    else:
        verts_np = verts

    verts_min = verts_np.min(axis=0)
    verts_max = verts_np.max(axis=0)
    extent = verts_max - verts_min

    bounds_min = verts_min - extent * padding
    bounds_max = verts_max + extent * padding

    return bounds_min.astype(np.float32), bounds_max.astype(np.float32)


def compute_probe_bounds_from_cameras(c2w_poses, padding=2.0):
    """
    Compute probe volume bounds from camera positions.

    Better for interior scenes (like Sponza) where mesh includes thick walls
    but cameras only see the interior.

    Args:
        c2w_poses: (N, 4, 4) or (N, 3, 4) camera-to-world matrices
        padding: Padding in world units (default 2.0 meters)

    Returns:
        bounds_min: (3,) minimum corner
        bounds_max: (3,) maximum corner
    """
    # Extract camera positions (translation column)
    cam_positions = c2w_poses[:, :3, 3]  # (N, 3)

    pos_min = cam_positions.min(axis=0)
    pos_max = cam_positions.max(axis=0)

    bounds_min = pos_min - padding
    bounds_max = pos_max + padding

    return bounds_min.astype(np.float32), bounds_max.astype(np.float32)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_root",
        type=str,
        default=os.path.join("data", "custom", "MyNeRFData"),
    )
    parser.add_argument(
        "--obj_name",
        type=str,
        default="sponza_gt.obj",
    )
    parser.add_argument(
        "--transforms",
        type=str,
        default="transforms_train.json",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="uv_lookup.npz",
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
        "--legacy_xy_flip",
        action="store_true",
    )
    parser.add_argument(
        "--probe_bounds_from_cameras",
        action="store_true",
        help="Compute probe bounds from camera positions instead of mesh vertices. "
             "Better for interior scenes like Sponza where mesh includes thick walls.",
    )
    parser.add_argument(
        "--probe_padding",
        type=float,
        default=None,
        help="Padding for probe bounds. If --probe_bounds_from_cameras: padding in world units (default 2.0). "
             "Otherwise: fractional padding (default 0.1 = 10%%).",
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

    print("UV raster device:", device)
    print("torch.cuda.is_available:", torch.cuda.is_available())

    data = load_blender_cameras(args.data_root, args.transforms)
    hwf = data["images_hwf"]
    c2w = data["c2w"]
    image_paths = data["image_paths"]

    ds = max(1, int(args.downscale))
    if ds > 1:
        hwf = hwf.copy()
        hwf[0] = hwf[0] / ds
        hwf[1] = hwf[1] / ds
        hwf[2] = hwf[2] / ds

    obj_path = os.path.join(args.data_root, args.obj_name)
    if not os.path.exists(obj_path):
        raise RuntimeError("OBJ file not found: {}".format(obj_path))

    verts, faces, aux = load_obj(obj_path, device=device, load_textures=False)
    verts_uvs = aux.verts_uvs
    faces_uvs = faces.textures_idx

    if verts_uvs is None or faces_uvs is None or verts_uvs.numel() == 0 or faces_uvs.numel() == 0:
        raise RuntimeError("OBJ file has no valid UV coordinates.")

    faces_verts = faces.verts_idx
    mesh = Meshes(verts=[verts], faces=[faces_verts]).to(device)

    cameras = build_cameras(c2w, hwf, device, legacy_xy_flip=args.legacy_xy_flip)

    uv, mask, pos = compute_uv_lookup(
        mesh, cameras, hwf, device, verts_uvs, faces_uvs, verts, faces_verts
    )

    # Compute probe bounds
    if args.probe_bounds_from_cameras:
        # Use camera positions - better for interior scenes
        padding = args.probe_padding if args.probe_padding is not None else 2.0
        probe_bounds_min, probe_bounds_max = compute_probe_bounds_from_cameras(c2w, padding=padding)
        print(f"Probe bounds (from cameras, padding={padding}): min={probe_bounds_min}, max={probe_bounds_max}")
    else:
        # Use mesh vertices - default
        padding = args.probe_padding if args.probe_padding is not None else 0.1
        probe_bounds_min, probe_bounds_max = compute_probe_bounds(verts, padding=padding)
        print(f"Probe bounds (from mesh, padding={padding*100:.0f}%): min={probe_bounds_min}, max={probe_bounds_max}")

    print(f"Position map range: min={pos[mask].min(axis=0)}, max={pos[mask].max(axis=0)}")

    output_path = os.path.join(args.data_root, args.output)
    np.savez_compressed(
        output_path,
        uv=uv.astype(np.float32),
        mask=mask.astype(np.bool_),
        pos=pos.astype(np.float32),
        probe_bounds_min=probe_bounds_min,
        probe_bounds_max=probe_bounds_max,
        image_paths=np.array(image_paths),
        H=np.int32(hwf[0]),
        W=np.int32(hwf[1]),
        focal=np.float32(hwf[2]),
    )


if __name__ == "__main__":
    main()

