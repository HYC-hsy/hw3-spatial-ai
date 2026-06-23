# HW3 深度学习与空间智能

本仓库用于完成期末作业的两个项目：

1. 基于 3DGS 与 AIGC 的多源资产生成与真实场景融合
2. 基于 LeRobot 的 ACT 策略跨环境泛化挑战

代码默认按服务器路径配置：

```bash
/mnt/data/kw/hy/projects/course/DL/hw3
```

本地保存代码、报告模板和数据集占位目录；服务器运行时使用同样的相对结构，数据统一放在 `hw3/data/` 下。

## 同步到服务器

本地代码目录：

```bash
/Users/tomato/Documents/potato/project/course/DL/hw3
```

服务器运行目录：

```bash
/mnt/data/kw/hy/projects/course/DL/hw3
```

建议在服务器上创建同名目录后，将本仓库内容同步过去：

```bash
rsync -av --exclude .git /Users/tomato/Documents/potato/project/course/DL/hw3/ \
  user@server:/mnt/data/kw/hy/projects/course/DL/hw3/
```

如果代码已经在服务器路径中，所有脚本都可以直接运行。若临时放在别处，可以设置：

```bash
export HW3_CODE_ROOT=/path/to/hw3
export HW3_ROOT=/mnt/data/kw/hy/projects/course/DL/hw3
```

后续所有命令默认在项目根目录运行：

```bash
cd /mnt/data/kw/hy/projects/course/DL/hw3
```

## 目录

```text
configs/                    # 路径、训练、融合配置
scripts/                    # 可直接复制运行的入口脚本
src/hw3_spatial_ai/          # Python 工具包
report/                     # LaTeX 报告模板和结果表格目录
assets/                     # 自采物体图片/视频占位目录
```

## 环境配置

推荐在服务器上运行：

```bash
cd /mnt/data/kw/hy/projects/course/DL/hw3
conda env create -f environment.yml
conda activate hw3-spatial-ai
pip install -e .
```

如果没有执行 `pip install -e .`，也可以在项目根目录临时使用 `PYTHONPATH=src`。例如：

```bash
cd /mnt/data/kw/hy/projects/course/DL/hw3
PYTHONPATH=src python -m hw3_spatial_ai.calvin.inspect_dataset \
  --data-root data/calvin_task_ABC_D
```

如果需要安装外部项目：

```bash
bash scripts/setup_external_repos.sh
```

该脚本会准备：

- `external/gaussian-splatting`
- `external/threestudio`
- `external/zero123`
- `external/lerobot`

不同服务器 CUDA/驱动版本可能不同，安装 3DGS 子模块时如遇编译失败，请按报错调整 PyTorch/CUDA 版本。

## 数据准备

### CALVIN

服务器路径：

```bash
/mnt/data/kw/hy/projects/course/DL/hw3/data/calvin_task_ABC_D
```

本地对齐路径：

```bash
/Users/tomato/Documents/potato/project/course/DL/hw3/data/calvin_task_ABC_D
```

先检查数据结构：

```bash
PYTHONPATH=src python -m hw3_spatial_ai.calvin.inspect_dataset \
  --data-root data/calvin_task_ABC_D
```

如果数据已经是 LeRobotDataset 格式，脚本会识别 `meta/` 和 `data/`。如果是 Hugging Face Arrow/parquet 格式，先运行：

```bash
PYTHONPATH=src python -m hw3_spatial_ai.calvin.prepare_lerobot_splits \
  --data-root data/calvin_task_ABC_D \
  --output-root processed/calvin_lerobot
```

该脚本会生成三个 split：

- `env_a_train`
- `env_abc_train`
- `env_d_eval`

### Mip-NeRF 360

服务器路径：

```bash
/mnt/data/kw/hy/projects/course/DL/hw3/data/360_extra_scenes
```

本地参考路径：

```bash
/Users/tomato/Documents/potato/project/course/DL/hw3/data/360_extra_scenes
```

当前参考目录包含 `flowers` 和 `treehill`。推荐先使用 `flowers`，因为它已经包含 `images/`、多尺度 `images_2/4/8/` 和 `sparse/` COLMAP 结构。确认目录中包含图像后运行：

```bash
PYTHONPATH=src python -m hw3_spatial_ai.scene.find_mipnerf_scene \
  --data-root data/360_extra_scenes \
  --scene flowers
```

## 题目二：ACT 训练与测试

### 训练环境 A 基础策略

```bash
bash scripts/train_act_env_a.sh
```

等价 Python 命令：

```bash
PYTHONPATH=src python -m hw3_spatial_ai.lerobot.launch_act \
  --config configs/act_env_a.yaml
```

### 训练 A/B/C 联合策略

```bash
bash scripts/train_act_env_abc.sh
```

### 在环境 D 零样本测试

```bash
bash scripts/eval_act_env_d.sh outputs/act_env_a/checkpoints/best
bash scripts/eval_act_env_d.sh outputs/act_env_abc/checkpoints/best
```

训练日志会写入：

```text
/mnt/data/kw/hy/projects/course/DL/hw3/outputs
```

导出对比表：

```bash
PYTHONPATH=src python -m hw3_spatial_ai.lerobot.summarize_runs \
  --runs-root /mnt/data/kw/hy/projects/course/DL/hw3/outputs \
  --out report/tables/act_results.csv
```

## 题目一：3DGS 与多源资产融合

### 背景场景 3DGS

```bash
bash scripts/run_3dgs_background.sh flowers
```

### 物体 A：真实多视角重建

把手机拍摄的视频或多视角照片放在：

```text
assets/object_a/
```

然后运行：

```bash
bash scripts/run_object_a_3dgs.sh object_a
```

### 物体 B：文本到 3D

修改 `configs/text_to_3d.yaml` 中的 prompt，然后：

```bash
bash scripts/run_threestudio_text3d.sh
```

### 物体 C：单图到 3D

把去背景后的单张图片放到：

```text
assets/object_c/input.png
```

然后：

```bash
bash scripts/run_zero123_image3d.sh assets/object_c/input.png
```

### Blender 融合渲染

把 A/B/C 的 `ply` 或 `obj` 路径填入 `configs/fusion_scene.yaml` 后：

```bash
PYTHONPATH=src python -m hw3_spatial_ai.scene.write_blender_fusion \
  --config configs/fusion_scene.yaml \
  --out /mnt/data/kw/hy/projects/course/DL/hw3/outputs/fusion/fuse_and_render.py

blender --background --python /mnt/data/kw/hy/projects/course/DL/hw3/outputs/fusion/fuse_and_render.py
```

## 报告

报告模板位于：

```text
report/main.tex
```

编译：

```bash
cd report
latexmk -pdf main.tex
```

报告中需要补充：

- GitHub 仓库链接
- 模型权重网盘链接
- WandB/SwanLab 导出的 loss 曲线
- ACT 成功率/动作误差表
- 3DGS/AIGC 三类资产在几何、纹理、耗时上的对比
