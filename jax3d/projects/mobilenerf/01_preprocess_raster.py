import os
import json
import argparse

import numpy as np
from PIL import Image

import torch
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.renderer import (
    PerspectiveCameras,
    RasterizationSettings,
    MeshRasterizer,
)
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


def build_cameras(c2w, hwf, device):
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

    w2c = torch.inverse(c2w_torch)
    R = w2c[:, :3, :3]
    T = w2c[:, :3, 3]

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


def compute_uv_lookup(mesh, cameras, hwf, device, faces_per_pixel=1):
    h, w, _ = hwf.astype(np.int32)
    raster_settings = RasterizationSettings(
        image_size=(int(h), int(w)),
        blur_radius=0.0,
        faces_per_pixel=faces_per_pixel,
        bin_size=0,
    )

    rasterizer = MeshRasterizer(raster_settings=raster_settings)

    textures = mesh.textures
    verts_uvs = textures.verts_uvs_packed()
    faces_uvs = textures.faces_uvs_packed()

    all_uv = []
    all_mask = []

    num_views = cameras.R.shape[0]
    for i in tqdm(range(num_views), desc="Rasterizing views", unit="view"):
        cam = cameras[i : i + 1]
        fragments = rasterizer(meshes_world=mesh, cameras=cam)
        pix_to_face = fragments.pix_to_face[0, ..., 0]
        bary_coords = fragments.bary_coords[0, ..., 0, :]

        pix_to_face_flat = pix_to_face.view(-1)
        bary_flat = bary_coords.view(-1, 3)

        valid = pix_to_face_flat >= 0
        face_idx_valid = pix_to_face_flat[valid]

        face_uv_idx = faces_uvs[face_idx_valid]
        verts_uv = verts_uvs[face_uv_idx]

        bary_valid = bary_flat[valid].unsqueeze(-1)
        uv_valid = (verts_uv * bary_valid).sum(dim=1)

        uv = torch.zeros(
            pix_to_face_flat.shape[0],
            2,
            dtype=torch.float32,
            device=device,
        )
        uv[valid] = uv_valid

        uv = uv.view(int(h), int(w), 2)
        mask = valid.view(int(h), int(w))

        all_uv.append(uv.cpu().numpy())
        all_mask.append(mask.cpu().numpy())

    all_uv = np.stack(all_uv, axis=0)
    all_mask = np.stack(all_mask, axis=0)

    return all_uv, all_mask


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
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data = load_blender_cameras(args.data_root, args.transforms)
    hwf = data["images_hwf"]
    c2w = data["c2w"]
    image_paths = data["image_paths"]

    obj_path = os.path.join(args.data_root, args.obj_name)
    if not os.path.exists(obj_path):
        raise RuntimeError("OBJ file not found: {}".format(obj_path))

    mesh = load_objs_as_meshes([obj_path], device=device)

    cameras = build_cameras(c2w, hwf, device)

    uv, mask = compute_uv_lookup(mesh, cameras, hwf, device)

    output_path = os.path.join(args.data_root, args.output)
    np.savez_compressed(
        output_path,
        uv=uv.astype(np.float32),
        mask=mask.astype(np.bool_),
        image_paths=np.array(image_paths),
        H=np.int32(hwf[0]),
        W=np.int32(hwf[1]),
        focal=np.float32(hwf[2]),
    )


if __name__ == "__main__":
    main()

