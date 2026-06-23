# Task 1 Status

Root: `/mnt/data/kw/hy/projects/course/DL/hw3`

## Installed real toolchain

- COLMAP: available in `.conda_envs/colmap_blender`
- Blender: `external/bin/blender` -> Blender 4.2.0
- threestudio/SDS/Zero123: available in `.conda_envs/threestudio`
- CUDA extensions installed: `tinycudann`, `nvdiffrast`, `envlight`
- Caches/downloads: project-local under `.cache/`

Check all tools:

```bash
cd /mnt/data/kw/hy/projects/course/DL/hw3
bash scripts/check_real_3d_env.sh
```

## Completed deliverables

- Object A point cloud: `outputs/object_a/point_cloud/iteration_30000/point_cloud.ply`
- Object B real SDS OBJ: `outputs/text_asset_b_real/export/model.obj`
- Object B 10K-step test video: `outputs/text_asset_b_real/object_b_sds/robot_prompt@20260619-144634/save/it10000-test.mp4`
- Object C real Zero123 OBJ: `outputs/image_asset_c_real/export/model.obj`
- Object C 600-step test video: `outputs/image_asset_c_real/object_c_zero123/chair_image@20260619-151900/save/it600-test.mp4`
- Background 3DGS PLY: `outputs/3dgs_background/flowers/point_cloud/iteration_30000/point_cloud.ply`
- Background renders: `outputs/3dgs_background/flowers/train/ours_30000/renders/`
- Fusion Blender scene: `outputs/fusion/fusion_scene.blend`
- Fusion video: `outputs/fusion/fusion_walkthrough.mp4`
- Report PDF: `report/main.pdf`

## Notes

- `external/threestudio/load/zero123/zero123-xl.ckpt` has been downloaded under the project root.
- Fusion currently uses the 30K Flowers background 3DGS + Object A + real SDS Object B + real Zero123 Object C.
- `metrics.py` wrote `results.json`; metrics are NaN because this Mip-NeRF 360 split has no held-out test cameras for that script, but training renders and the 30K PLY are present.

## Commands

Object B, text-to-3D SDS:

```bash
cd /mnt/data/kw/hy/projects/course/DL/hw3
GPU=0 bash scripts/run_threestudio_text3d.sh 10000
```

Object C, Zero123 image-to-3D:

```bash
cd /mnt/data/kw/hy/projects/course/DL/hw3
GPU=0 bash scripts/run_zero123_image3d.sh assets/object_c/input.png 600
```

Fusion render:

```bash
cd /mnt/data/kw/hy/projects/course/DL/hw3
bash scripts/fuse_scene_blender.sh
```
