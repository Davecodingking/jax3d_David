# -*- coding: utf-8 -*-


# ==============================================================================
# 可选参考块: 安装 Conda + 构建 Python 3.9 环境 + 克隆官方 jax3d（一般可以跳过）
# 说明: 仅当你需要对照官方原版仓库时再运行这一块；
#      常规训练建议直接从下方 Cell 1 开始，使用 jax3d_David 仓库。
# ==============================================================================
import os

print("⏳ 正在安装 Conda (可能需要 1-2 分钟)...")
!pip install -q condacolab
import condacolab
condacolab.install()

import time
time.sleep(5) # 给它一点时间反应
print("✅ Conda 安装完成！正在配置 Python 3.9 环境...")

# 创建环境并锁死 JAX 版本 (0.3.25)
!conda create -n mobilenerf python=3.9 -y
!source activate mobilenerf && conda install -c conda-forge cudatoolkit=11.8 cudnn=8.8 -y
# 使用 --no-deps 防止 pip 自动升级 JAX
!source activate mobilenerf && pip install "jax[cuda11_pip]==0.3.25" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html --no-deps
# 手动补齐 JAX 依赖
!source activate mobilenerf && pip install "flax==0.5.3" scipy "optax==0.1.4" "chex==0.1.5" "absl-py" --no-deps
# 安装其他工具
!source activate mobilenerf && pip install tqdm opencv-python-headless matplotlib gin-config msgpack typing-extensions opt_einsum toolz rich PyYAML numpy==1.23.5

# 下载代码
if not os.path.exists('/content/jax3d'):
    !git clone https://github.com/google-research/jax3d.git

print("✅ 环境搭建完毕！JAX 版本已锁死为 0.3.25")

"""# Cell 1: 一键配置环境 + 拉取 jax3d_David 仓库（推荐默认）"""

# ==============================================================================
# Cell 1: 一键配置环境 + 拉取 Dave 的修复版代码
# ==============================================================================
import os
import time
import shutil

# --- 1. 安装基础环境 (Conda) ---
print("⏳ 正在安装 Conda...")
try:
    import condacolab
except ImportError:
    !pip install -q condacolab
    import condacolab
condacolab.install()
time.sleep(5)

print("✅ Conda 就绪！配置 Python 3.9 + JAX...")

# --- 2. 配置 Python 环境 (锁死版本) ---
!conda create -n mobilenerf python=3.9 -y
# 安装 CUDA, JAX, Flax (必须严格匹配)
!source activate mobilenerf && conda install -c conda-forge cudatoolkit=11.8 cudnn=8.8 -y
!source activate mobilenerf && pip install "jax[cuda11_pip]==0.3.25" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html --no-deps
!source activate mobilenerf && pip install "flax==0.5.3" scipy "optax==0.1.4" "chex==0.1.5" "absl-py" --no-deps
!source activate mobilenerf && pip install tqdm opencv-python-headless matplotlib gin-config msgpack typing-extensions opt_einsum toolz rich PyYAML numpy==1.23.5

print("✅ 运行环境搭建完毕！")

# --- 3. 拉取你的 GitHub 代码 (jax3d_David) ---
# 你的仓库地址
MY_REPO = "https://github.com/Davecodingking/jax3d_David.git"
TARGET_DIR = "/content/jax3d"

print(f"🚀 正在拉取你的代码: {MY_REPO}")

# 清理旧目录
if os.path.exists(TARGET_DIR): shutil.rmtree(TARGET_DIR)
if os.path.exists("/content/jax3d_David"): shutil.rmtree("/content/jax3d_David")

# 克隆仓库
!git clone {MY_REPO}

# 结构修正: 把 jax3d_David 改名为 jax3d (Python 才能识别)
if os.path.exists("/content/jax3d_David"):
    shutil.move("/content/jax3d_David", TARGET_DIR)
    print("✅ 代码已就位 (jax3d_David -> jax3d)")
else:
    print("❌ 克隆失败，请检查 GitHub 仓库是否为空或地址错误。")

print("🎉 准备就绪！请上传数据包 MyNeRFData.zip 并运行下一步。")

"""# Cell 2: 打包当前 jax3d 仓库（可选）"""

import shutil
import os
from google.colab import files

# 1. 定义打包目标：整个 jax3d 仓库 (包含修好的 mobilenerf)
source_dir = '/content/jax3d'
output_filename = '/content/jax3d_fixed_final'

print(f"📦 正在打包修复后的代码库: {source_dir} ...")
print("   (这包含了 Scale=0.033, No-Gamma, RGB-Fix 的所有修改)")

# 2. 压缩
shutil.make_archive(output_filename, 'zip', source_dir)

print(f"✅ 打包完成: {output_filename}.zip")
print("⬇️ 请在左侧文件栏找到 'jax3d_fixed_final.zip'，右键下载并妥善保存！")
# files.download(output_filename + '.zip') # 你可以手动取消注释让它自动下载

# --- 3. 拉取你的 GitHub 代码 (jax3d_David) ---
# 你的仓库地址
MY_REPO = "https://github.com/Davecodingking/jax3d_David.git"
TARGET_DIR = "/content/jax3d"

print(f"🚀 正在拉取你的代码: {MY_REPO}")

# 清理旧目录
if os.path.exists(TARGET_DIR): shutil.rmtree(TARGET_DIR)
if os.path.exists("/content/jax3d_David"): shutil.rmtree("/content/jax3d_David")

# 克隆仓库
!git clone {MY_REPO}

# 结构修正: 把 jax3d_David 改名为 jax3d (Python 才能识别)
if os.path.exists("/content/jax3d_David"):
    shutil.move("/content/jax3d_David", TARGET_DIR)
    print("✅ 代码已就位 (jax3d_David -> jax3d)")
else:
    print("❌ 克隆失败，请检查 GitHub 仓库是否为空或地址错误。")

print("🎉 准备就绪！请上传数据包 MyNeRFData.zip 并运行下一步。")

"""# Cell 3: 数据准备（解压 Dataset + 结构修复 + 挂载 Drive）"""

# ==========================================
# 步骤 2 (修复版): 智能解压与完整性检查
# 说明：请保证 ZIP 内部顶层文件夹命名为 MyNeRFData，
#       且最终展开路径为 data/custom/MyNeRFData，对应 mobilenerf 中 object_name="MyNeRFData"
# ==========================================
import os
import zipfile

zip_path = '/content/MyNeRFData.zip'
extract_path = '/content/jax3d/jax3d/projects/mobilenerf/data/custom/MyNeRFData'

# 1. 检查文件是否存在
if not os.path.exists(zip_path):
    print("❌ 错误：根本没找到 /content/MyNeRFData.zip！")
    print("👉 请将文件拖入左侧文件栏，并等待上传完成。")
    assert False

# 2. 检查文件是否损坏 (关键步骤)
if not zipfile.is_zipfile(zip_path):
    print("❌ 致命错误：ZIP 文件已损坏或未上传完成！")
    print("💡 原因：通常是因为你在上传进度条走完之前就点击了运行。")
    print("👉 解决：请在左侧删除该文件，重新上传，务必等待下方进度圈完全消失！")
    # 打印文件大小看看
    file_size = os.path.getsize(zip_path) / (1024 * 1024)
    print(f"   当前文件大小仅为: {file_size:.2f} MB (如果这个数很小，说明肯定没传完)")
    assert False

print(f"✅ ZIP 文件完整 ({os.path.getsize(zip_path)/1024/1024:.2f} MB)。准备解压...")

# 3. 创建目录
if not os.path.exists(extract_path):
    os.makedirs(extract_path)

# 4. 解压
print(f"📂 正在解压到: {extract_path}")
# 使用 python 自带库解压，比 shell 命令更稳
try:
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    print("✅ 解压成功！")
except Exception as e:
    print(f"❌ 解压失败: {e}")
    assert False

# 5. 再次核实内容
# 有时候 zip 包里自带了一层文件夹，我们需要确认 json 在哪
print("🧐 核实文件位置...")
found_json = False
for root, dirs, files in os.walk(extract_path):
    if "transforms_train.json" in files:
        print(f"✅ 成功找到配置文件: {os.path.join(root, 'transforms_train.json')}")
        found_json = True
        break

if not found_json:
    print("⚠️ 警告：解压成功，但没找到 transforms_train.json。")
    print("👇 请检查下面的文件结构，看看是不是套了一层文件夹？")
    for root, dirs, files in os.walk(extract_path):
        level = root.replace(extract_path, '').count(os.sep)
        indent = ' ' * 4 * (level)
        print('{}{}/'.format(indent, os.path.basename(root)))
        subindent = ' ' * 4 * (level + 1)
        for f in files[:5]: # 只显示前5个文件
            print('{}{}'.format(subindent, f))
else:
    print("🎉 数据准备完美就绪！请继续运行 Step 3。")

import os
import shutil

nested_dir = '/content/jax3d/jax3d/projects/mobilenerf/data/custom/MyNeRFData/MyNeRFData'
target_dir = '/content/jax3d/jax3d/projects/mobilenerf/data/custom/MyNeRFData'

print("🔧 正在检测是否套娃...")

if os.path.exists(nested_dir):
    print(f"⚠️ 发现套娃文件夹！正在把文件从 {nested_dir} 搬出来...")

    # 遍历套娃文件夹里的所有文件，移动到外面
    for filename in os.listdir(nested_dir):
        src = os.path.join(nested_dir, filename)
        dst = os.path.join(target_dir, filename)

        # 如果目标已存在，先删除，防止报错
        if os.path.exists(dst):
            if os.path.isdir(dst): shutil.rmtree(dst)
            else: os.remove(dst)

        shutil.move(src, dst)
        print(f"  -> 移动: {filename}")

    # 删掉空的套娃壳子
    os.rmdir(nested_dir)
    print("✅ 搬家完成！结构已修复。")
else:
    print("ℹ️ 没发现套娃文件夹。")
    # 检查一下文件到底在哪
    if os.path.exists(os.path.join(target_dir, "transforms_train.json")):
        print("✅ 确认：json 文件已经在正确位置了。")
    else:
        print("❌ 依然找不到文件，请检查左侧文件栏确认路径。")

import os
import time
import threading
import shutil
import re
from google.colab import drive
from IPython import get_ipython

# 1. 环境准备
if not os.path.exists('/content/drive'):
    drive.mount('/content/drive')

project_dir = '/content/jax3d/jax3d/projects/mobilenerf'
if os.path.exists(project_dir):
    os.chdir(project_dir)
    os.environ['PYTHONPATH'] += ":/content/jax3d"
else:
    print(f"❌ 找不到项目目录: {project_dir}")





"""# Cell 4: Stage1 备份 + Stage2 代码修复入口"""

# ==============================================================================
# Stage 2 强制修复脚本 (无论如何都要把 0.033 塞进去！)
# ==============================================================================
import os

PROJECT_ROOT = "/content/jax3d/jax3d/projects/mobilenerf"
target_file = os.path.join(PROJECT_ROOT, 'stage2.py')
LOCAL_SAMPLES_DIR = os.path.join(PROJECT_ROOT, "samples_stage2")
# 你的 Drive 输出路径
DRIVE_OUTPUT_DIR = "/content/drive/MyDrive/Stage1_12Jan/Stage2_Result_256"

print(f"🔧 正在暴力修复 {target_file} ...")

with open(target_file, 'r') as f:
    lines = f.readlines()

new_lines = []
scale_injected = False

for line in lines:
    # 1. 强制修改分辨率为 256
    if "point_grid_size =" in line and "128" in line:
        line = "    point_grid_size = 256 # [Force 256]\n"
        print("   ✅ 分辨率已强制改为 256")

    # 2. 强制修改 Samples 路径
    if "os.path.join(base_dir, 'samples')" in line:
        line = line.replace("os.path.join(base_dir, 'samples')", f"'{LOCAL_SAMPLES_DIR}'")
    if "os.path.join(logdir, 'samples')" in line:
        line = line.replace("os.path.join(logdir, 'samples')", f"'{LOCAL_SAMPLES_DIR}'")

    # 3. 🔥 核心：在 return 之前强制插入 Scale 0.033
    # 我们寻找这一行，一旦找到，就在它前面插队
    if "return {'images' : images" in line and not scale_injected:
        print("   🔥 [重要] 正在注入 Scale 0.033 代码...")
        # 写入缩放逻辑
        new_lines.append("\n    # [FORCE INJECTED SCALE]\n")
        new_lines.append("    print('⚡⚡⚡ APPLYING SCALE 0.033 ⚡⚡⚡')\n")
        new_lines.append("    poses = poses.at[:, :3, 3].set(poses[:, :3, 3] * 0.033)\n")
        new_lines.append(line) # 把原来的 return 写回去
        scale_injected = True
        continue

    # 4. 修复 Stage 1 权重读取路径 (防止读不到)
    if "pickle.load" in line and "weights_stage1.pkl" in line:
        line = "    vars = pickle.load(open('weights/weights_stage1.pkl', 'rb'))\n"
        print("   ✅ 权重读取路径已修正")

    new_lines.append(line)

# 写回文件
with open(target_file, 'w') as f:
    f.writelines(new_lines)

if scale_injected:
    print("\n🎉 修复成功！0.033 缩放代码已强制写入。")
else:
    print("\n❌ 严重错误：没找到注入点！请检查 stage2.py 内容。")

"""# Cell 5: 简易版 Stage1 启动（备份 + 训练）"""

# ==============================================================================
# 🛡️ Stage1 Step 1: 强盗备份脚本
# ==============================================================================
local_checkpoints = "checkpoints"
local_weights = "weights"
local_samples = "samples"

drive_root = "/content/drive/MyDrive/Stage1_12Jan"
drive_checkpoints = os.path.join(drive_root, "checkpoints")
drive_weights = os.path.join(drive_root, "weights")
drive_samples = os.path.join(drive_root, "samples")

def background_backup():
    print("🛡️ 后台备份服务已启动...")
    while True:
        try:
            if os.path.exists(local_checkpoints):
                if not os.path.exists(drive_checkpoints): os.makedirs(drive_checkpoints)
                os.system(f"cp -ru '{local_checkpoints}/.' '{drive_checkpoints}/'")
            if os.path.exists(local_weights):
                if not os.path.exists(drive_weights): os.makedirs(drive_weights)
                os.system(f"cp -ru '{local_weights}/.' '{drive_weights}/'")
            if os.path.exists(local_samples):
                if not os.path.exists(drive_samples): os.makedirs(drive_samples)
                os.system(f"cp -ru '{local_samples}/.' '{drive_samples}/'")
        except:
            pass
        time.sleep(30)

t = threading.Thread(target=background_backup)
t.daemon = True
t.start()

# ==============================================================================
# 🚀 Stage1 Step 2: 启动训练
# ==============================================================================
print("🚀 启动训练...")

cmd = """
source activate mobilenerf && export MPLBACKEND=Agg && python stage1.py \
  --gin_configs=configs/360.gin \
  --gin_bindings="Config.dataset_loader='blender'" \
  --gin_bindings="Config.batch_size=4096" \
  --gin_bindings="Config.data_dir='/content/jax3d/jax3d/projects/mobilenerf/data/custom/MyNeRFData'" \
  --gin_bindings="Config.checkpoint_dir='/content/jax3d/jax3d/projects/mobilenerf/checkpoints'" \
  --gin_bindings="Config.render_every=1000" \
  --gin_bindings="Config.save_every=2000" \
  --logtostderr
"""

get_ipython().system(cmd)

"""# Cell 6: Stage2 适配 .pkl 存档 + Scale 0.033"""

# ==============================================================================
# 🛡️ Stage 2 复刻修正版 (修复路径 + Scale 0.033 + Res 256)
# ==============================================================================
import os
import re
import pickle
import shutil
import glob
from google.colab import drive
from IPython import get_ipython

# --- 1. 路径与参数配置 ---
# Drive 源头 (Stage 1 权重)
DRIVE_SOURCE_PKL_DIR = "/content/drive/MyDrive/Stage1_12Jan/weights"
# Drive 输出 (Stage 2 结果) - 直接存这里防断连
DRIVE_OUTPUT_DIR = "/content/drive/MyDrive/Stage1_12Jan/Stage2_Result_256"

# 本地路径
PROJECT_ROOT = "/content/jax3d/jax3d/projects/mobilenerf"
LOCAL_WEIGHTS_DIR = os.path.join(PROJECT_ROOT, "weights")
LOCAL_SAMPLES_DIR = os.path.join(PROJECT_ROOT, "samples_stage2")

# 关键参数
TARGET_SCALE = 0.033
TARGET_GRID = 256

# ==============================================================================
# Stage2 [Step 1] 环境准备
# ==============================================================================
print("🚚 [1/4] 环境准备中...")

if not os.path.exists('/content/drive'): drive.mount('/content/drive')
if not os.path.exists(LOCAL_WEIGHTS_DIR): os.makedirs(LOCAL_WEIGHTS_DIR)
if not os.path.exists(LOCAL_SAMPLES_DIR): os.makedirs(LOCAL_SAMPLES_DIR)
if not os.path.exists(DRIVE_OUTPUT_DIR): os.makedirs(DRIVE_OUTPUT_DIR)

# 搬运权重 (确保 weights/weights_stage1.pkl 存在)
print(f"    📥 正在查找 Stage 1 权重...")
pkl_files = glob.glob(os.path.join(DRIVE_SOURCE_PKL_DIR, "*.pkl"))
if not pkl_files:
    pkl_files = glob.glob(os.path.join(os.path.dirname(DRIVE_SOURCE_PKL_DIR), "*.pkl"))

if pkl_files:
    pkl_files.sort(key=os.path.getmtime)
    target_pkl = pkl_files[-1]
    shutil.copy2(target_pkl, os.path.join(LOCAL_WEIGHTS_DIR, "weights_stage1.pkl"))
    print(f"      -> 已就位: {os.path.basename(target_pkl)}")
else:
    print("❌ 错误：Drive 里没找到 .pkl 文件！")
    assert False

# ==============================================================================
# Stage2 [Step 2] 代码手术（路径与采样配置）
# ==============================================================================
target_file = os.path.join(PROJECT_ROOT, 'stage2.py')
print(f"💉 [2/4] 修改代码...")

with open(target_file, 'r') as f:
    content = f.read()

# 1. 基础修复
content = re.sub(r'object_name = "chair"', 'object_name = "MyNeRFData"', content)
content = re.sub(r'scene_dir = "datasets/nerf_synthetic/.*?\+object_name', 'scene_dir = "data/custom/"+object_name', content)
content = content.replace("import matplotlib.pyplot as plt", "import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt")

# 3. 修正 Samples 输出路径
content = content.replace("os.path.join(base_dir, 'samples')", f"'{LOCAL_SAMPLES_DIR}'")
content = content.replace("os.path.join(logdir, 'samples')", f"'{LOCAL_SAMPLES_DIR}'")

# 4. [关键修改] 升级分辨率到 256
if "point_grid_size = 128" in content:
    content = content.replace("point_grid_size = 128", f"point_grid_size = {TARGET_GRID} # [A100 Force]")
    print(f"    ✅ 分辨率已修改为 {TARGET_GRID}")

# 5. [关键修改] 注入 Scale 0.033
original_return = "return {'images' : images, 'c2w' : poses, 'hwf' : hwf}"
if "poses[:, :3, 3] * " + str(TARGET_SCALE) not in content:
    injection_code = f"""
    # [Auto-Scale Injection]
    print("⚡ Applying Scale {TARGET_SCALE}...")
    poses = poses.at[:, :3, 3].set(poses[:, :3, 3] * {TARGET_SCALE})
    {original_return}
    """
    content = content.replace(original_return, injection_code)
    print(f"    ✅ Scale {TARGET_SCALE} 注入代码已插入")

# 6. 修复 Stage 1 权重读取 (确保读 weights/weights_stage1.pkl)
# 原代码可能是 pickle.load(open(weights_dir+"/"+"weights_stage1.pkl", "rb"))
# 我们直接硬改
if 'weights_stage1.pkl' in content:
    # 这里的正则稍微宽泛一点，匹配 open(...) 里的内容
    content = re.sub(r'open\(.*?"weights_stage1\.pkl".*?,', 'open("weights/weights_stage1.pkl",', content)

with open(target_file, 'w') as f:
    f.write(content)

# ==============================================================================
# Stage2 [Step 3] 启动 Stage 2
# ==============================================================================
print(f"\n🚀 [3/4] 启动 Stage 2 ...")
print(f"    💾 存档将直接写入: {DRIVE_OUTPUT_DIR}")

os.chdir(PROJECT_ROOT)

# 启动命令
cmd = f"""
source activate mobilenerf && export MPLBACKEND=Agg && python stage2.py \
  --gin_configs=configs/360.gin \
  --gin_bindings="Config.dataset_loader='blender'" \
  --gin_bindings="Config.batch_size=2048" \
  --gin_bindings="Config.data_dir='data/custom/MyNeRFData'" \
  --gin_bindings="Config.checkpoint_dir='{DRIVE_OUTPUT_DIR}'" \
  --logtostderr
"""

get_ipython().system(cmd)

# ==============================================================================
# Stage2 [Step 4] 额外备份 Samples
# ==============================================================================
print("\n📦 [4/4] 备份 Samples 图片...")
if os.path.exists(LOCAL_SAMPLES_DIR):
    drive_sample_dest = DRIVE_OUTPUT_DIR + "_samples"
    if not os.path.exists(drive_sample_dest): os.makedirs(drive_sample_dest)
    os.system(f"cp -r '{LOCAL_SAMPLES_DIR}/.' '{drive_sample_dest}/'")

"""# Cell 7: Stage3 提取 Mesh 与纹理（适配 Drive 读取 + 独立输出）"""

# ==============================================================================
# 🏛️ Stage 3 最终版（接力权重导出 Mesh + 纹理）
# ==============================================================================
import os
import re
import shutil
from google.colab import drive
from IPython import get_ipython

# --- 1. 路径配置 ---
PROJECT_ROOT = "/content/jax3d/jax3d/projects/mobilenerf"

# 关键：读取刚才 Stage 2 刚生成的新权重 (本地)
LOCAL_S2_WEIGHTS_READ = os.path.join(PROJECT_ROOT, "weights")

# 输出路径
LOCAL_S3_OBJ_SAVE = os.path.join(PROJECT_ROOT, "obj_stage3_256")
LOCAL_S3_SAMPLES_SAVE = os.path.join(PROJECT_ROOT, "samples_stage3_256")

# Drive 备份路径
DRIVE_FINAL_EXPORT = "/content/drive/MyDrive/Stage1_12Jan/Final_Sponza_256"

# ================= Step 1: 环境准备 & 权重检查 =================
print("🚚 [1/4] 正在检查权重...")

# 确保输出目录存在
if not os.path.exists(LOCAL_S3_OBJ_SAVE): os.makedirs(LOCAL_S3_OBJ_SAVE)
if not os.path.exists(LOCAL_S3_SAMPLES_SAVE): os.makedirs(LOCAL_S3_SAMPLES_SAVE)

# 检查本地是否有刚才 Stage 2 跑出来的权重
if not os.path.exists(LOCAL_S2_WEIGHTS_READ):
    print(f"❌ 错误：找不到 Stage 2 的输出目录: {LOCAL_S2_WEIGHTS_READ}")
    print("   请确保刚才的 Stage 2 已经成功跑完！")
    assert False

# 找最新的权重文件
pkl_files = [f for f in os.listdir(LOCAL_S2_WEIGHTS_READ) if f.endswith(".pkl")]
if not pkl_files:
    print(f"❌ 错误：在 {LOCAL_S2_WEIGHTS_READ} 里没找到 .pkl 文件！")
    assert False

# 排序找到最新的 (通常是 checkpoint_20000 这种)
pkl_files.sort(key=lambda x: int(re.search(r'\d+', x).group()) if re.search(r'\d+', x) else 0)
target_pkl = pkl_files[-1]
print(f"   ✅ 锁定 Stage 2 权重: {target_pkl}")

# ⚠️ 关键动作：把这个文件复制一份并在原地改名为 'weights_stage2_1.pkl'
# 因为 Stage 3 代码里通常写死读这个名字，或者读 list 的最后一个
# 为了稳妥，我们手动帮它准备好
dest_pkl = os.path.join(LOCAL_S2_WEIGHTS_READ, "weights_stage2_1.pkl")
shutil.copy2(os.path.join(LOCAL_S2_WEIGHTS_READ, target_pkl), dest_pkl)
print(f"   📦 已准备好标准命名权重: weights_stage2_1.pkl")


# ================= Step 2: 代码精准手术 =================
print("\n💉 [2/4] 修改 Stage 3 代码 (分辨率256 + 盒子0.7)...")

target_file = os.path.join(PROJECT_ROOT, 'stage3.py')
with open(target_file, 'r') as f:
    content = f.read()

# A. 路径修复
content = content.replace('object_name = "chair"', 'object_name = "MyNeRFData"')
content = content.replace('scene_dir = "datasets/nerf_synthetic/"+object_name', 'scene_dir = "data/custom/"+object_name')
# 指向我们刚准备好的本地目录
content = content.replace('weights_dir = "weights"', f'weights_dir = "{LOCAL_S2_WEIGHTS_READ}"')
content = content.replace('obj_save_dir = "obj"', f'obj_save_dir = "{LOCAL_S3_OBJ_SAVE}"')
content = content.replace('samples_dir = "samples"', f'samples_dir = "{LOCAL_S3_SAMPLES_SAVE}"')

# B. 🔥 关键同步：分辨率必须是 256 (匹配 Stage 2)
if "point_grid_size = 128" in content:
    content = content.replace("point_grid_size = 128", "point_grid_size = 256 # [MATCH STAGE 2]")
    print("   ✅ 分辨率已同步为 256 (匹配权重)")
elif "point_grid_size = 256" in content:
    print("   ✅ 分辨率保持 256")

# C. 🔥 关键优化：盒子大小 Scale 0.7 (适配 1.22m Sponza)
# 1.4m 的盒子装 1.22m 的物体，利用率极高
if "scene_grid_scale = 1.2" in content:
    content = content.replace("scene_grid_scale = 1.2", "scene_grid_scale = 0.7 # [Fit Sponza]")
    print("   ✅ 捕获框优化为 0.7 (紧凑高精)")
elif "scene_grid_scale = 0.2" in content:
    content = content.replace("scene_grid_scale = 0.2", "scene_grid_scale = 0.7 # [Fit Sponza]")
    print("   ✅ 捕获框修正为 0.7")

# D. JAX Scale 注入 (保持 0.033)
if "poses = np.stack(cams, axis=0)" in content:
    content = content.replace("poses[:, :3, 3] *= 0.033", "")
    if ".at[" not in content:
        content = content.replace(
            "poses = np.stack(cams, axis=0)",
            "poses = np.stack(cams, axis=0)\n    poses = poses.at[:, :3, 3].set(poses[:, :3, 3] * 0.033) # [JAX Scale]"
        )

# E. 移除 Testing (加速)
if 'print("Testing")' in content:
    content = content.replace('print("Testing")', '# print("Testing")')
    content = content.replace('for i in tqdm(range(len(data[\'test\'][\'images\']))):', 'for i in []:\n  pass')
    content = content.replace('for p in tqdm(render_poses):', 'for p in []:\n  pass')

with open(target_file, 'w') as f:
    f.write(content)

# ================= Step 3: 启动 =================
print("\n🚀 [3/4] 启动 Stage 3 (High Res Export)...")
os.chdir(PROJECT_ROOT)
cmd = "source activate mobilenerf && export MPLBACKEND=Agg && python stage3.py"

try:
    get_ipython().system(cmd)
    print("\n🎉 Stage 3 提取结束！")
except Exception as e:
    print(f"\n❌ 运行出错: {e}")

# ================= Step 4: 备份 =================
print("\n📦 [4/4] 备份最终结果到 Drive...")
if not os.path.exists('/content/drive'): drive.mount('/content/drive')
if not os.path.exists(DRIVE_FINAL_EXPORT): os.makedirs(DRIVE_FINAL_EXPORT)

if os.path.exists(LOCAL_S3_OBJ_SAVE):
    print(f"   -> 正在备份 OBJ/GLB 到: {DRIVE_FINAL_EXPORT}")
    os.system(f"cp -r '{LOCAL_S3_OBJ_SAVE}/.' '{DRIVE_FINAL_EXPORT}/'")

    # 备份 Phone 版
    phone_src = LOCAL_S3_OBJ_SAVE + "_phone"
    if os.path.exists(phone_src):
        phone_dst = os.path.join(DRIVE_FINAL_EXPORT, "obj_phone")
        if not os.path.exists(phone_dst): os.makedirs(phone_dst)
        os.system(f"cp -r '{phone_src}/.' '{phone_dst}/'")

print(f"\n✅ 全部完成！高精度 Sponza 已保存至: {DRIVE_FINAL_EXPORT}")

"""# Cell 8: Mesh-Based UV Preprocessing (预计算 像素→UV 映射)"""

import os
from google.colab import drive
from IPython import get_ipython

PROJECT_ROOT = "/content/jax3d/jax3d/projects/mobilenerf"
UV_OUTPUT = os.path.join(PROJECT_ROOT, "data/custom/MyNeRFData/mesh_uv_samples.npz")
DRIVE_TEXTURE_EXPORT = "/content/drive/MyDrive/Stage1_12Jan/TextureTraining"

print("🚚 [1/3] 准备环境并挂载 Drive...")
if not os.path.exists("/content/drive"):
    drive.mount("/content/drive")

if os.path.exists(PROJECT_ROOT):
    os.chdir(PROJECT_ROOT)
    os.environ["PYTHONPATH"] += ":/content/jax3d"
else:
    print(f"❌ 找不到项目目录: {PROJECT_ROOT}")

print("🔧 [2/3] 预计算 像素→UV 映射 (preprocess_mesh_uv.py)...")
cmd = "source activate mobilenerf && export MPLBACKEND=Agg && python preprocess_mesh_uv.py"
get_ipython().system(cmd)

if os.path.exists(UV_OUTPUT):
    print(f"   ✅ 已生成 UV 采样文件: {UV_OUTPUT}")
else:
    print("   ⚠️ 未在预期路径找到 mesh_uv_samples.npz，请检查上方日志输出。")

print("📦 [3/3] 备份 UV 采样文件到 Drive...")
if not os.path.exists(DRIVE_TEXTURE_EXPORT):
    os.makedirs(DRIVE_TEXTURE_EXPORT)

uv_dst = os.path.join(DRIVE_TEXTURE_EXPORT, "uv_samples")
if not os.path.exists(uv_dst):
    os.makedirs(uv_dst)

if os.path.exists(UV_OUTPUT):
    os.system(f"cp '{UV_OUTPUT}' '{uv_dst}/'")

print(f"🎉 UV 预处理完成，结果已备份至: {uv_dst}")

"""# Cell 9: Mesh-Based Neural Texture Training (训练特征纹理 + 预览导出)"""

import os
from google.colab import drive
from IPython import get_ipython

PROJECT_ROOT = "/content/jax3d/jax3d/projects/mobilenerf"
DRIVE_TEXTURE_EXPORT = "/content/drive/MyDrive/Stage1_12Jan/TextureTraining"

print("🚚 [1/3] 准备环境并挂载 Drive...")
if not os.path.exists("/content/drive"):
    drive.mount("/content/drive")

if os.path.exists(PROJECT_ROOT):
    os.chdir(PROJECT_ROOT)
    os.environ["PYTHONPATH"] += ":/content/jax3d"
else:
    print(f"❌ 找不到项目目录: {PROJECT_ROOT}")

print("🚀 [2/3] 正在训练特征纹理 (train_texture.py)...")
cmd = "source activate mobilenerf && export MPLBACKEND=Agg && python train_texture.py"
get_ipython().system(cmd)

TEXTURES_DIR = os.path.join(PROJECT_ROOT, "textures")
WEIGHTS_DIR = os.path.join(PROJECT_ROOT, "weights_texture")

print("📦 [3/3] 备份纹理、权重与预览图到 Drive...")
if not os.path.exists(DRIVE_TEXTURE_EXPORT):
    os.makedirs(DRIVE_TEXTURE_EXPORT)

textures_dst = os.path.join(DRIVE_TEXTURE_EXPORT, "textures")
weights_dst = os.path.join(DRIVE_TEXTURE_EXPORT, "weights")

if os.path.exists(TEXTURES_DIR):
    if not os.path.exists(textures_dst):
        os.makedirs(textures_dst)
    os.system(f"cp -r '{TEXTURES_DIR}/.' '{textures_dst}/'")

if os.path.exists(WEIGHTS_DIR):
    if not os.path.exists(weights_dst):
        os.makedirs(weights_dst)
    os.system(f"cp -r '{WEIGHTS_DIR}/.' '{weights_dst}/'")

print(f"🎉 Mesh-Based Neural Texture 训练完成，纹理、权重与预览图已备份至: {DRIVE_TEXTURE_EXPORT}")
