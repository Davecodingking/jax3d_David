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
        grid = uv.view(1, b, 1, 2)
        feat = F.grid_sample(
            self.texture,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        feat = feat.squeeze(0).squeeze(-1).permute(1, 0).contiguous()
        x = torch.cat([feat, viewdirs], dim=-1)
        rgb = torch.sigmoid(self.mlp(x))
        return rgb  


def export_mobilenerf_assets(model, texture_size, export_dir, mesh_path=None):
    """
    Export assets in MobileNeRF Unity Viewer compatible format.

    Expected by Unity viewer (julienkay/MobileNeRF-Unity-Viewer):
    - shape0.obj (mesh)
    - shape0.pngfeat0.png, shape0.pngfeat1.png (feature textures)
    - mlp.json (MLP weights)
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
    }

    scene_params_path = os.path.join(export_dir, "mlp.json")
    with open(scene_params_path, "w", encoding="utf-8") as f:
        json.dump(mlp_params, f)

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
    ds = max(1, int(args.downscale))
    if ds > 1:
        uv_np = uv_np[:, ::ds, ::ds, :]
        mask_np = mask_np[:, ::ds, ::ds]
    uv_all = torch.from_numpy(uv_np).to(device=device, dtype=torch.float32)
    mask_all = torch.from_numpy(mask_np).to(device=device)

    train_data = load_blender_train(args.data_root, args.transforms)
    images_np = train_data["images"]
    c2w_np = train_data["c2w"]
    hwf = train_data["hwf"]

    if ds > 1:
        images_np = images_np[:, ::ds, ::ds, :]
        hwf[0] = hwf[0] / ds
        hwf[1] = hwf[1] / ds
        hwf[2] = hwf[2] / ds

    images = torch.from_numpy(images_np).to(device=device, dtype=torch.float32)
    c2w_all = torch.from_numpy(c2w_np).to(device=device, dtype=torch.float32)

    h, w, focal = hwf
    pix2cam = pix2cam_matrix(h, w, focal, device)

    model = HybridTextureMLP(texture_size=args.texture_size).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    start_step = 0
    
    # -------------------------------------------------------------------------
    # Resume from checkpoint if available
    # -------------------------------------------------------------------------
    if os.path.exists(args.checkpoint):
        print(f"🔄 Found checkpoint: {args.checkpoint}. Resuming training...")
        try:
            checkpoint_data = torch.load(args.checkpoint, map_location=device)
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
        )
        if batch is None:
            continue

        uv_grid, viewdirs, gt_rgb = batch

        optimizer.zero_grad()
        pred_rgb = model(uv_grid, viewdirs)
        loss = criterion(pred_rgb, gt_rgb)
        loss.backward()
        optimizer.step()

        t = (step + 1) / float(args.num_iters)
        lr_now = args.lr * (0.01 ** t)
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

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "step": step + 1,
                    "hwf": hwf,
                },
                args.checkpoint,
            )

            export_mobilenerf_assets(model, args.texture_size, args.export_dir, args.mesh_path)

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
