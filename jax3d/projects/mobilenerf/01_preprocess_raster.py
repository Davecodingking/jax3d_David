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


def compute_uv_lookup(mesh, cameras, hwf, device, verts_uvs, faces_uvs, faces_per_pixel=1):
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
            mask = valid.view(int(h), int(w))
            uv = uv.view(int(h), int(w), 2)
        else:
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

    mesh = Meshes(verts=[verts], faces=[faces.verts_idx]).to(device)

    cameras = build_cameras(c2w, hwf, device, legacy_xy_flip=args.legacy_xy_flip)

    uv, mask = compute_uv_lookup(mesh, cameras, hwf, device, verts_uvs, faces_uvs)

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

