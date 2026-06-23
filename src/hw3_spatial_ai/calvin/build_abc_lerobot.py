
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')


def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    try:
        os.link(src, dst)
    except OSError:
        try:
            dst.symlink_to(src.resolve())
        except OSError:
            shutil.copy2(src, dst)


def add_nested(value, delta):
    if isinstance(value, list):
        return [add_nested(v, delta) for v in value]
    if isinstance(value, (int, float)):
        return value + delta
    return value


def adjust_episode_stats(stats: dict, new_ep: int, frame_delta: int, task_delta: int) -> dict:
    out = json.loads(json.dumps(stats))
    out['episode_index'] = new_ep
    s = out.get('stats', {})
    for key, delta in [('episode_index', new_ep - stats['episode_index']), ('index', frame_delta), ('task_index', task_delta), ('annotation.human.action.task_description', task_delta)]:
        if key in s:
            for field in ['min', 'max', 'mean']:
                if field in s[key]:
                    s[key][field] = add_nested(s[key][field], delta)
    return out


def combine_stats(stats_paths: list[Path]) -> dict:
    # Weighted merge of top-level stats.json entries. This dataset version has
    # no per-key count field, so use each shard's total_frames as the weight.
    raw = []
    weights = []
    for p in stats_paths:
        raw.append(json.loads(p.read_text()))
        info = json.loads((p.parent / 'info.json').read_text())
        weights.append(float(info['total_frames']))
    keys = raw[0].keys()
    merged = {}
    total_weight = sum(weights)
    for key in keys:
        means = [np.asarray(item[key]['mean'], dtype=np.float64) for item in raw]
        stds = [np.asarray(item[key]['std'], dtype=np.float64) for item in raw]
        mins = [np.asarray(item[key]['min'], dtype=np.float64) for item in raw]
        maxs = [np.asarray(item[key]['max'], dtype=np.float64) for item in raw]
        mean = sum(w * m for w, m in zip(weights, means)) / total_weight
        second = sum(w * (sd ** 2 + m ** 2) for w, sd, m in zip(weights, stds, means)) / total_weight
        var = np.maximum(second - mean ** 2, 0)
        out = {
            'mean': mean.tolist(),
            'std': np.sqrt(var).tolist(),
            'min': np.minimum.reduce(mins).tolist(),
            'max': np.maximum.reduce(maxs).tolist(),
        }
        if all('q01' in item[key] for item in raw):
            out['q01'] = np.minimum.reduce([np.asarray(item[key]['q01'], dtype=np.float64) for item in raw]).tolist()
        if all('q99' in item[key] for item in raw):
            out['q99'] = np.maximum.reduce([np.asarray(item[key]['q99'], dtype=np.float64) for item in raw]).tolist()
        merged[key] = out
    return merged


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', default='data/calvin_task_ABC_D')
    parser.add_argument('--output-root', default='processed/calvin_lerobot/env_abc_train')
    parser.add_argument('--parts', nargs='+', default=['0', '1', '2'])
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()

    data_root = Path(args.data_root)
    out = Path(args.output_root)
    if out.exists() and args.force:
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    episode_offset = 0
    frame_offset = 0
    task_offset = 0
    all_episodes = []
    all_episode_stats = []
    all_tasks = []
    info_total_frames = 0
    info_total_videos = 0
    stats_paths = []
    template_info = None

    for part in args.parts:
        src = data_root / f'calvin_task_ABC_D_lerobot_{part}_4'
        meta = src / 'meta'
        info = json.loads((meta / 'info.json').read_text())
        if template_info is None:
            template_info = info
        stats_paths.append(meta / 'stats.json')
        tasks = read_jsonl(meta / 'tasks.jsonl')
        episodes = read_jsonl(meta / 'episodes.jsonl')
        ep_stats = read_jsonl(meta / 'episodes_stats.jsonl')

        for t in tasks:
            nt = dict(t)
            nt['task_index'] = t['task_index'] + task_offset
            all_tasks.append(nt)

        ep_stats_by_old = {e['episode_index']: e for e in ep_stats}
        for ep in episodes:
            old_ep = ep['episode_index']
            new_ep = old_ep + episode_offset
            old_chunk = old_ep // info['chunks_size']
            new_chunk = new_ep // info['chunks_size']
            src_parquet = src / info['data_path'].format(episode_chunk=old_chunk, episode_index=old_ep)
            dst_parquet = out / info['data_path'].format(episode_chunk=new_chunk, episode_index=new_ep)
            dst_parquet.parent.mkdir(parents=True, exist_ok=True)

            df = pd.read_parquet(src_parquet)
            n = len(df)
            df = df.copy()
            df['episode_index'] = new_ep
            df['index'] = np.arange(frame_offset, frame_offset + n, dtype=np.int64)
            if 'task_index' in df:
                df['task_index'] = df['task_index'] + task_offset
            if 'annotation.human.action.task_description' in df:
                df['annotation.human.action.task_description'] = df['annotation.human.action.task_description'] + task_offset
            df.to_parquet(dst_parquet, index=False)

            for video_key in ['observation.images.image', 'observation.images.wrist_image']:
                src_video = src / info['video_path'].format(episode_chunk=old_chunk, video_key=video_key, episode_index=old_ep)
                dst_video = out / info['video_path'].format(episode_chunk=new_chunk, video_key=video_key, episode_index=new_ep)
                link_or_copy(src_video, dst_video)

            nep = dict(ep)
            nep['episode_index'] = new_ep
            all_episodes.append(nep)
            all_episode_stats.append(adjust_episode_stats(ep_stats_by_old[old_ep], new_ep, frame_offset, task_offset))
            frame_offset += n

        episode_offset += info['total_episodes']
        task_offset += info['total_tasks']
        info_total_frames += info['total_frames']
        info_total_videos += info['total_videos']

    info = dict(template_info)
    info['total_episodes'] = episode_offset
    info['total_frames'] = info_total_frames
    info['total_tasks'] = task_offset
    info['total_videos'] = info_total_videos
    info['total_chunks'] = (episode_offset + info['chunks_size'] - 1) // info['chunks_size']
    info['splits'] = {'train': f'0:{episode_offset}'}

    (out / 'meta').mkdir(parents=True, exist_ok=True)
    (out / 'meta/info.json').write_text(json.dumps(info, ensure_ascii=False, indent=4), encoding='utf-8')
    (out / 'meta/stats.json').write_text(json.dumps(combine_stats(stats_paths), ensure_ascii=False, indent=4), encoding='utf-8')
    # modality is structural and shared.
    shutil.copy2(data_root / 'calvin_task_ABC_D_lerobot_0_4/meta/modality.json', out / 'meta/modality.json')
    write_jsonl(out / 'meta/tasks.jsonl', all_tasks)
    write_jsonl(out / 'meta/episodes.jsonl', all_episodes)
    write_jsonl(out / 'meta/episodes_stats.jsonl', all_episode_stats)
    print(f'Wrote {out}')
    print(f'episodes={episode_offset} frames={info_total_frames} tasks={task_offset} chunks={info["total_chunks"]}')


if __name__ == '__main__':
    main()
