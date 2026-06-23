# HW3: 深度学习与空间智能

> 课程期末作业 | 2026 春

## 项目概述

本项目包含两个子任务：

| 任务 | 内容 | 关键技术 |
|------|------|----------|
| Task 1 | 基于 3DGS 与 AIGC 的多源资产生成与真实场景融合 | 3D Gaussian Splatting, SDS (Score Distillation Sampling), Zero123, Blender |
| Task 2 | 基于 LeRobot 的 ACT 策略跨环境泛化挑战 | Action Chunking with Transformers (ACT), CALVIN Benchmark |

## 实验结果

### Task 1: 3DGS 场景融合

通过三种不同方式生成 3D 资产，并融合到 3DGS 重建的真实背景场景中：

- **Object A**：多视角照片 → COLMAP + 3DGS 重建（30K iterations）
- **Object B**：文本描述 → threestudio SDS 生成（10K steps）
- **Object C**：单张图片 → Zero123 生成（600 steps）
- **背景场景**：Mip-NeRF 360 Flowers 数据集 → 3DGS 重建
- **融合渲染**：Blender 合成最终场景漫游视频

### Task 2: ACT 跨环境泛化

在 CALVIN 仿真平台上训练 ACT 策略，评估从训练环境到未见环境的零样本迁移能力：

- 环境 A 单独训练 → 环境 D 零样本测试
- 环境 A/B/C 联合训练 → 环境 D 零样本测试

训练曲线和评估指标见 `report/figures/` 和 `report/tables/`。

## 仓库结构

```
configs/             # 训练与融合配置文件
scripts/             # 可执行的入口脚本
src/hw3_spatial_ai/  # Python 源码包
├── calvin/          # 数据集检查与预处理
├── lerobot/         # ACT 训练、评估、结果汇总
└── scene/           # 3DGS 场景处理与 Blender 融合
report/              # 实验报告（LaTeX 源码 + PDF + 图表）
assets/              # 输入素材（物体照片等）
```

## 环境配置

```bash
conda env create -f environment.yml
conda activate hw3-spatial-ai
pip install -e .
```

外部依赖（3DGS、threestudio、Zero123、LeRobot）安装：

```bash
bash scripts/setup_external_repos.sh
```

## 运行说明

### Task 1: 3D 资产生成与融合

```bash
# 背景场景 3DGS 重建
bash scripts/run_3dgs_background.sh flowers

# Object A: 多视角重建
bash scripts/run_object_a_3dgs.sh object_a

# Object B: 文本生成 3D
bash scripts/run_threestudio_text3d.sh 10000

# Object C: 单图生成 3D
bash scripts/run_zero123_image3d.sh assets/object_c/input.png 600

# Blender 融合渲染
bash scripts/fuse_scene_blender.sh
```

### Task 2: ACT 训练与评估

```bash
# 环境 A 基础策略训练
bash scripts/train_act_env_a.sh

# 环境 A/B/C 联合策略训练
bash scripts/train_act_env_abc.sh

# 环境 D 零样本评估
bash scripts/eval_act_env_d.sh outputs/act_env_a/checkpoints/best
bash scripts/eval_act_env_d.sh outputs/act_env_abc/checkpoints/best
```

## 模型权重

预训练权重下载（1.04GB）：

> https://pan.baidu.com/s/1ca0_krZEDyvtKcsgTa6qvQ?pwd=7294
>
> 提取码: `7294`

## 报告

完整实验报告见 [`report/main.pdf`](report/main.pdf)，包含：

- 三类 3D 资产在几何质量、纹理细节、生成耗时上的对比分析
- ACT 策略的训练曲线、成功率与动作误差对比
- 跨环境泛化的消融实验与分析

## 技术栈

- Python 3.10 | PyTorch 2.x | CUDA 12.x
- [3D Gaussian Splatting](https://github.com/graphdeco-inria/gaussian-splatting)
- [threestudio](https://github.com/threestudio-project/threestudio) (SDS / Zero123)
- [LeRobot](https://github.com/huggingface/lerobot) (ACT Policy)
- [CALVIN](https://github.com/mees/calvin) (Benchmark)
- Blender 4.2 (Scene Composition)
