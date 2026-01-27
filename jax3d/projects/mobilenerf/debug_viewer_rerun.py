import os

# 定义对比脚本内容
diagnostic_script = r'''
import os
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from pytorch3d.io import load_obj
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    PerspectiveCameras, RasterizationSettings, MeshRasterizer,
    MeshRenderer, SoftPhongShader, AmbientLights, TexturesVertex
)

# ================= CONFIG =================
DATA_ROOT = "data/custom/MyNeRFData"
OBJ_NAME = "sponza_gt.obj"
TRANSFORMS = "transforms_train.json"
DEVICE = "cuda"
TARGET_FRAMES = [0, 29] # 第 1 帧 和 第 30 帧
# ==========================================

def run_comparison():
    print(f"🚀 启动结构对比诊断: 检查 Index {TARGET_FRAMES}...")

    # 1. 加载模型
    obj_path = os.path.join(DATA_ROOT, OBJ_NAME)
    verts, faces, _ = load_obj(obj_path)
    verts = verts.to(DEVICE).float()
    faces_idx = faces.verts_idx.to(DEVICE).long()
    
    # 法线着色：观察几何轮廓
    temp_mesh = Meshes(verts=[verts], faces=[faces_idx])
    normals = temp_mesh.verts_normals_packed()
    verts_rgb = (normals + 1.0) / 2.0 
    textures = TexturesVertex(verts_features=verts_rgb[None])
    mesh = Meshes(verts=[verts], faces=[faces_idx], textures=textures)

    # 2. 加载元数据
    with open(os.path.join(DATA_ROOT, TRANSFORMS), "r") as f:
        meta = json.load(f)
    
    camera_angle_x = float(meta["camera_angle_x"])
    lights = AmbientLights(device=DEVICE, ambient_color=((1.0, 1.0, 1.0),))

    results = []

    for idx in TARGET_FRAMES:
        frame = meta["frames"][idx]
        img_path = os.path.join(DATA_ROOT, frame["file_path"] + ".png")
        if not os.path.exists(img_path):
            img_path = img_path.replace(".png", ".jpg")
        
        gt_image = np.array(Image.open(img_path).convert("RGB")) / 255.0
        H, W, _ = gt_image.shape

        c2w = np.array(frame["transform_matrix"], dtype=np.float32)
        c2w_t = torch.from_numpy(c2w).to(DEVICE)
        
        R_c2w = c2w_t[:3, :3]
        C = c2w_t[:3, 3]
        R_w2c = R_c2w.t()
        T_w2c = -R_w2c @ C
        
        focal = 0.5 * W / np.tan(0.5 * camera_angle_x)
        cameras = PerspectiveCameras(
            device=DEVICE,
            R=R_w2c.t().unsqueeze(0), # PyTorch3D 内部使用行向量乘法，所以传转置
            T=T_w2c.unsqueeze(0),
            focal_length=((focal, focal),),
            principal_point=((W/2, H/2),),
            image_size=((H, W),),
            in_ndc=False
        )

        raster_settings = RasterizationSettings(image_size=(H, W), blur_radius=0.0, faces_per_pixel=1, cull_backfaces=False)
        renderer = MeshRenderer(
            rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
            shader=SoftPhongShader(device=DEVICE, cameras=cameras, lights=lights)
        )
        
        render_rgba = renderer(mesh)[0].cpu().numpy()
        render_rgb = render_rgba[..., :3]
        results.append((gt_image, render_rgb, idx))

    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    for i, (gt, ren, idx) in enumerate(results):
        axes[i, 0].imshow(gt)
        axes[i, 0].set_title(f"Frame {idx}: Unity Photo", fontsize=16)
        axes[i, 0].axis("off")
        
        axes[i, 1].imshow(ren)
        axes[i, 1].set_title(f"Frame {idx}: Pytorch3D Render (Raw)", fontsize=16)
        axes[i, 1].axis("off")
    
    plt.tight_layout()
    output_name = "frame_comparison_result.png"
    plt.savefig(output_name)
    print(f"✅ 对比图已生成: {output_name}")

if __name__ == "__main__":
    run_comparison()
'''

# 写入文件
with open("compare_script.py", "w") as f:
    f.write(diagnostic_script)

# 执行诊断
print("🚀 正在运行对比诊断...")
!source activate hybrid_uv_wheel && export MPLBACKEND=Agg && python compare_script.py

# 显示结果
from IPython.display import Image, display
if os.path.exists("frame_comparison_result.png"):
    display(Image("frame_comparison_result.png"))
