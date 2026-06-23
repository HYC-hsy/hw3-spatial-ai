from __future__ import annotations

import argparse
from pathlib import Path

from hw3_spatial_ai.config import ensure_dir, load_yaml


BLENDER_TEMPLATE = r'''
import math
from pathlib import Path

import bpy

OUTPUT_DIR = Path(r"{output_dir}")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete()

scene = bpy.context.scene
scene.render.resolution_x = {width}
scene.render.resolution_y = {height}
scene.frame_start = 1
scene.frame_end = {frames}
scene.render.fps = {fps}
scene.render.image_settings.file_format = "FFMPEG"
scene.render.ffmpeg.format = "MPEG4"
scene.render.ffmpeg.codec = "H264"
scene.render.filepath = str(OUTPUT_DIR / "fusion_walkthrough.mp4")

def import_asset(path, name, location, rotation, scale):
    path = Path(path)
    if not path.exists():
        print(f"Missing asset: {{path}}")
        return None
    if path.suffix.lower() == ".obj":
        bpy.ops.wm.obj_import(filepath=str(path))
    elif path.suffix.lower() == ".ply":
        bpy.ops.wm.ply_import(filepath=str(path))
    else:
        raise ValueError(f"Unsupported asset format: {{path}}")
    objs = bpy.context.selected_objects
    empty = bpy.data.objects.new(name, None)
    bpy.context.collection.objects.link(empty)
    for obj in objs:
        obj.name = f"{{name}}_{{obj.name}}"
        obj.parent = empty
    empty.location = location
    empty.rotation_euler = tuple(math.radians(v) for v in rotation)
    empty.scale = scale
    return empty

assets = {assets}
for asset in assets:
    import_asset(asset["path"], asset["name"], asset["location"], asset["rotation"], asset["scale"])

bpy.ops.object.light_add(type="AREA", location=(0, -3, 4))
light = bpy.context.object
light.name = "soft_key_light"
light.data.energy = 500
light.data.size = 4

bpy.ops.object.camera_add(location=(0, -4, 2.0), rotation=(math.radians(64), 0, 0))
camera = bpy.context.object
scene.camera = camera

for frame in range(scene.frame_start, scene.frame_end + 1):
    t = (frame - scene.frame_start) / max(scene.frame_end - scene.frame_start, 1)
    angle = 2 * math.pi * t
    radius = 4.0
    camera.location = (radius * math.sin(angle), -radius * math.cos(angle), 2.0)
    direction = mathutils.Vector((0, 0, 0.4)) - camera.location
    camera.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
    camera.keyframe_insert(data_path="location", frame=frame)
    camera.keyframe_insert(data_path="rotation_euler", frame=frame)

bpy.ops.wm.save_as_mainfile(filepath=str(OUTPUT_DIR / "fusion_scene.blend"))
bpy.ops.render.render(animation=True)
'''


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    out = Path(args.out)
    ensure_dir(out.parent)
    output_dir = cfg["output_dir"]
    res = cfg.get("resolution", {})
    assets = cfg.get("objects", [])
    if cfg.get("background", {}).get("gaussian_ply"):
        bg = cfg["background"]
        assets = [
            {
                "name": f"background_{bg.get('name', 'scene')}",
                "path": bg["gaussian_ply"],
                "location": [0, 0, 0],
                "rotation": [0, 0, 0],
                "scale": [1, 1, 1],
            },
            *assets,
        ]

    script = BLENDER_TEMPLATE.format(
        output_dir=output_dir,
        width=res.get("width", 1280),
        height=res.get("height", 720),
        frames=cfg.get("frames", 180),
        fps=cfg.get("fps", 30),
        assets=repr(assets),
    )
    script = "import mathutils\n" + script
    out.write_text(script, encoding="utf-8")
    print(f"Wrote Blender script: {out}")


if __name__ == "__main__":
    main()

