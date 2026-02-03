# ============================================================
# 渲染对比测试 - 验证 unwrapped mesh 几何正确性
# ============================================================

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

# 配置
DATA_ROOT = "data/custom/MyNeRFData"
ORIGINAL_OBJ = "sponza_gt.obj"
UNWRAPPED_OBJ = "sponza_gt_unwarpped.obj"

# 检查 Drive 备份
DRIVE_ROOT = "/content/drive/MyDrive/Hybrid_Pipeline"
if not os.path.exists(os.path.join(DATA_ROOT, UNWRAPPED_OBJ)):
    drive_obj = os.path.join(DRIVE_ROOT, UNWRAPPED_OBJ)
    if os.path.exists(drive_obj):
        print(f"🔄 从 Drive 恢复 {UNWRAPPED_OBJ}...")
        os.system(f"cp '{drive_obj}' '{DATA_ROOT}/'")

TRANSFORMS = "transforms_train.json"
TEST_FRAME = 0  # 测试第一帧
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("="*60)
print("🎨 渲染对比测试")
print("="*60)
print(f"Device: {DEVICE}")

# ============================================================
# 加载元数据
# ============================================================
print("\n[1/5] 加载相机参数...")

with open(os.path.join(DATA_ROOT, TRANSFORMS), "r") as f:
    meta = json.load(f)

frame = meta["frames"][TEST_FRAME]
camera_angle_x = float(meta["camera_angle_x"])

# 加载 ground truth 图片
img_path = os.path.join(DATA_ROOT, frame["file_path"] + ".png")
if not os.path.exists(img_path):
    img_path = img_path.replace(".png", ".jpg")

gt_image = np.array(Image.open(img_path).convert("RGB")) / 255.0
H, W, _ = gt_image.shape
print(f"   图片尺寸: {W}x{H}")

# 相机矩阵
c2w = np.array(frame["transform_matrix"], dtype=np.float32)
c2w_t = torch.from_numpy(c2w).to(DEVICE)

R_c2w = c2w_t[:3, :3]
C = c2w_t[:3, 3]
R_w2c = R_c2w.t()
T_w2c = -R_w2c @ C

focal = 0.5 * W / np.tan(0.5 * camera_angle_x)

cameras = PerspectiveCameras(
    device=DEVICE,
    R=R_w2c.t().unsqueeze(0),
    T=T_w2c.unsqueeze(0),
    focal_length=((focal, focal),),
    principal_point=((W/2, H/2),),
    image_size=((H, W),),
    in_ndc=False
)

print(f"   相机焦距: {focal:.2f}")

# ============================================================
# 渲染函数
# ============================================================
def render_mesh(obj_name, title):
    """渲染 mesh"""
    print(f"\n渲染 {title}...")
    
    obj_path = os.path.join(DATA_ROOT, obj_name)
    verts, faces, _ = load_obj(obj_path)
    verts = verts.to(DEVICE).float()
    faces_idx = faces.verts_idx.to(DEVICE).long()
    
    print(f"   顶点: {len(verts):,}, 面: {len(faces_idx):,}")
    
    # 使用法线着色（便于观察几何）
    temp_mesh = Meshes(verts=[verts], faces=[faces_idx])
    normals = temp_mesh.verts_normals_packed()
    verts_rgb = (normals + 1.0) / 2.0
    textures = TexturesVertex(verts_features=verts_rgb[None])
    
    mesh = Meshes(verts=[verts], faces=[faces_idx], textures=textures)
    
    # 渲染设置
    lights = AmbientLights(device=DEVICE, ambient_color=((1.0, 1.0, 1.0),))
    raster_settings = RasterizationSettings(
        image_size=(H, W),
        blur_radius=0.0,
        faces_per_pixel=1,
        cull_backfaces=False
    )
    
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        shader=SoftPhongShader(device=DEVICE, cameras=cameras, lights=lights)
    )
    
    render_rgba = renderer(mesh)[0].cpu().numpy()
    render_rgb = render_rgba[..., :3]
    
    return render_rgb

# ============================================================
# 渲染两个 mesh
# ============================================================
print("\n[2/5] 渲染原始 mesh...")
try:
    render_original = render_mesh(ORIGINAL_OBJ, "Original")
    has_original = True
except Exception as e:
    print(f"   ⚠️  渲染失败: {e}")
    has_original = False

print("\n[3/5] 渲染 unwrapped mesh...")
try:
    render_unwrapped = render_mesh(UNWRAPPED_OBJ, "Unwrapped")
    has_unwrapped = True
except Exception as e:
    print(f"   ❌ 渲染失败: {e}")
    has_unwrapped = False

# ============================================================
# 对比可视化
# ============================================================
print("\n[4/5] 生成对比图...")

if has_original and has_unwrapped:
    # 3x2 布局
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 第一行：原始 mesh
    axes[0, 0].imshow(render_original)
    axes[0, 0].set_title("Original Mesh Render", fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')
    
    axes[0, 1].imshow(render_unwrapped)
    axes[0, 1].set_title("Unwrapped Mesh Render", fontsize=12, fontweight='bold')
    axes[0, 1].axis('off')
    
    diff_between = np.abs(render_original - render_unwrapped).mean(axis=2)
    im1 = axes[0, 2].imshow(diff_between, cmap='hot', vmin=0, vmax=0.5)
    axes[0, 2].set_title("Difference (Original - Unwrapped)", fontsize=12, fontweight='bold')
    axes[0, 2].axis('off')
    plt.colorbar(im1, ax=axes[0, 2], fraction=0.046)
    
    # 第二行：unwrapped mesh
    axes[1, 0].imshow(gt_image)
    axes[1, 0].set_title("Ground Truth (Unity Photo)", fontsize=12, fontweight='bold')
    axes[1, 0].axis('off')
    
    diff_original = np.abs(gt_image - render_original).mean(axis=2)
    axes[1, 1].imshow(diff_original, cmap='hot', vmin=0, vmax=0.5)
    axes[1, 1].set_title("Difference (GT - Original)", fontsize=12, fontweight='bold')
    axes[1, 1].axis('off')
    
    diff_unwrapped = np.abs(gt_image - render_unwrapped).mean(axis=2)
    im2 = axes[1, 2].imshow(diff_unwrapped, cmap='hot', vmin=0, vmax=0.5)
    axes[1, 2].set_title("Difference (GT - Unwrapped)", fontsize=12, fontweight='bold')
    axes[1, 2].axis('off')
    plt.colorbar(im2, ax=axes[1, 2], fraction=0.046)
    
    plt.tight_layout()
    plt.savefig('render_comparison.png', dpi=150, bbox_inches='tight')
    print("   ✅ 已保存: render_comparison.png")
    
elif has_unwrapped:
    # 只有 unwrapped
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    axes[0].imshow(gt_image)
    axes[0].set_title("Ground Truth", fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    axes[1].imshow(render_unwrapped)
    axes[1].set_title("Unwrapped Mesh Render", fontsize=12, fontweight='bold')
    axes[1].axis('off')
    
    diff = np.abs(gt_image - render_unwrapped).mean(axis=2)
    im = axes[2].imshow(diff, cmap='hot')
    axes[2].set_title("Difference", fontsize=12, fontweight='bold')
    axes[2].axis('off')
    plt.colorbar(im, ax=axes[2])
    
    plt.tight_layout()
    plt.savefig('render_comparison.png', dpi=150, bbox_inches='tight')
    print("   ✅ 已保存: render_comparison.png")

# ============================================================
# 定量分析
# ============================================================
print("\n[5/5] 定量分析...")

if has_original and has_unwrapped:
    mse_between = np.mean((render_original - render_unwrapped) ** 2)
    psnr_between = -10 * np.log10(mse_between) if mse_between > 0 else float('inf')

    mse_original = np.mean((gt_image - render_original) ** 2)
    mse_unwrapped = np.mean((gt_image - render_unwrapped) ** 2)

    psnr_original = -10 * np.log10(mse_original) if mse_original > 0 else float('inf')
    psnr_unwrapped = -10 * np.log10(mse_unwrapped) if mse_unwrapped > 0 else float('inf')

    print(f"\n原始 vs Unwrapped:")
    print(f"   MSE: {mse_between:.6f}")
    print(f"   PSNR: {psnr_between:.2f} dB")

    print(f"\n原始 mesh vs Ground Truth:")
    print(f"   MSE: {mse_original:.6f}")
    print(f"   PSNR: {psnr_original:.2f} dB")

    print(f"\nUnwrapped mesh vs Ground Truth:")
    print(f"   MSE: {mse_unwrapped:.6f}")
    print(f"   PSNR: {psnr_unwrapped:.2f} dB")

    if mse_between < 1e-4:
        print("\n   ✅ 原始与 unwrapped 渲染结果几乎一致")
        print("   ✅ UV unwrap 没有破坏几何")
    elif mse_between < 1e-3:
        print("\n   ✅ 原始与 unwrapped 渲染结果接近")
    else:
        print("\n   ⚠️  原始与 unwrapped 渲染有明显差异")

elif has_unwrapped:
    mse = np.mean((gt_image - render_unwrapped) ** 2)
    psnr = -10 * np.log10(mse) if mse > 0 else float('inf')
    
    print(f"\nUnwrapped mesh vs Ground Truth:")
    print(f"   MSE: {mse:.6f}")
    print(f"   PSNR: {psnr:.2f} dB")

# ============================================================
# 总结
# ============================================================
print("\n" + "="*60)
print("📊 渲染测试总结")
print("="*60)

if has_original and has_unwrapped:
    print("✅ 原始和 unwrapped mesh 都渲染成功")
    print("✅ 可以对比几何完整性")
elif has_unwrapped:
    print("✅ Unwrapped mesh 渲染成功")
    print("⚠️  原始 mesh 未找到，无法对比")
else:
    print("❌ 渲染失败")

print("\n查看 render_comparison.png 来判断:")
print("1. 两个 mesh 是否渲染一致")
print("2. 差异图是否接近黑色（差异小）")
print("3. 如果一致，说明 UV unwrap 成功保留了几何")
print("="*60)

# 显示结果
# 在非交互式环境中，不需要 display
try:
    from IPython.display import Image as IPImage, display
    if os.path.exists('render_comparison.png'):
        display(IPImage('render_comparison.png'))
except ImportError:
    pass # 忽略错误，因为图片已经保存了
