import os
import json
import argparse

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


def load_blender_train(data_dir, transforms_filename):
    with open(os.path.join(data_dir, transforms_filename), "r") as fp:
        meta = json.load(fp)

    cams = []
    image_paths = []
    for frame in meta["frames"]:
        cams.append(np.array(frame["transform_matrix"], dtype=np.float32))
        image_paths.append(os.path.join(data_dir, frame["file_path"] + ".png"))

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
    def __init__(self, texture_size, num_channels=8, hidden_features=(16, 16)):
        super().__init__()
        self.texture = nn.Parameter(
            torch.zeros(1, num_channels, texture_size, texture_size)
        )

        layers = []
        in_dim = num_channels + 3
        last_dim = in_dim
        for feat in hidden_features:
            layers.append(nn.Linear(last_dim, feat))
            layers.append(nn.ReLU(inplace=True))
            last_dim = feat
        layers.append(nn.Linear(last_dim, 3))
        self.mlp = nn.Sequential(*layers)

    def forward(self, uv, viewdirs):
        b = uv.shape[0]
        grid = uv.view(b, 1, 1, 2)
        tex = self.texture.expand(b, -1, -1, -1)
        feat = F.grid_sample(
            tex,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
        feat = feat.view(b, -1)
        x = torch.cat([feat, viewdirs], dim=-1)
        rgb = torch.sigmoid(self.mlp(x))
        return rgb


def sample_batch(
    batch_size,
    uv_all,
    mask_all,
    images,
    c2w_all,
    pix2cam,
    device,
):
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
        default=4096,
    )
    parser.add_argument(
        "--num_iters",
        type=int,
        default=200000,
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
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    uv_data = np.load(args.uv_path, allow_pickle=True)
    uv_all = torch.from_numpy(uv_data["uv"]).to(device=device, dtype=torch.float32)
    mask_all = torch.from_numpy(uv_data["mask"]).to(device=device)

    train_data = load_blender_train(args.data_root, args.transforms)
    images_np = train_data["images"]
    c2w_np = train_data["c2w"]
    hwf = train_data["hwf"]

    images = torch.from_numpy(images_np).to(device=device, dtype=torch.float32)
    c2w_all = torch.from_numpy(c2w_np).to(device=device, dtype=torch.float32)

    h, w, focal = hwf
    pix2cam = pix2cam_matrix(h, w, focal, device)

    model = HybridTextureMLP(texture_size=args.texture_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    os.makedirs(os.path.dirname(args.checkpoint), exist_ok=True)

    pbar = tqdm(range(args.num_iters), desc="Hybrid training", unit="iter")
    for step in pbar:
        batch = sample_batch(
            args.batch_size,
            uv_all,
            mask_all,
            images,
            c2w_all,
            pix2cam,
            device,
        )
        if batch is None:
            continue

        uv_grid, viewdirs, gt_rgb = batch

        optimizer.zero_grad()
        pred_rgb = model(uv_grid, viewdirs)
        loss = criterion(pred_rgb, gt_rgb)
        loss.backward()
        optimizer.step()

        if (step + 1) % 100 == 0:
            with torch.no_grad():
                mse = loss.item()
                psnr = -10.0 * np.log10(mse + 1e-8)
            pbar.set_postfix(loss=mse, psnr=psnr)

        if (step + 1) % 10000 == 0:
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "step": step + 1,
                    "hwf": hwf,
                },
                args.checkpoint,
            )

            view_idx = 0
            model.eval()
            with torch.no_grad():
                uv_view = uv_all[view_idx]
                h_img, w_img, _ = uv_view.shape
                uv_grid_full = uv_view * 2.0 - 1.0
                uv_flat = uv_grid_full.view(-1, 2)

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
                    preds.append(model(u, v))
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
