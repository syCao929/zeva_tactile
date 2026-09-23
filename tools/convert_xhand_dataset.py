#!/usr/bin/env python3
"""把 TactileTTT 的 LeRobot 数据集转成 Cosmos/Zeva 训练管线认识的目录布局。

设计要点
--------
* **只读源数据**：本脚本从不写入、移动、重命名源目录里的任何文件。
* **零拷贝**：data/ 与 videos/ 以软链方式引用源文件，因此不复制 36GB。
  源数据一旦移动，链接会失效——重跑本脚本即可修复。
* **不改语义**：相机映射（哪路充当 wrist）与 action/state 的维度选择
  由加载器 `xhand_lerobot_dataset.py` 决定，这里只做容器布局的搬运。

目标布局（与 RoboCasa365LeRobotDataset 的 `*/*/*/lerobot/meta/info.json` 约定一致）：

    <target>/<category>/<task_name>/<variant>/lerobot/
        meta/info.json          # 复制源文件
        meta/episodes.jsonl     # 复制源文件
        meta/tasks.jsonl        # 复制源文件
        meta/stats.json         # 复制源文件
        data/chunk-000          -> 源 data/chunk-000
        videos/chunk-000        -> 源 videos/chunk-000
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

# 源数据（TactileTTT 采集）在 zeva-work 之外，通过 --source 或 $XHAND_SOURCE 指定。
# 本机取值放在 env.local.sh（不入库），模板见 env.local.sh.example。
SOURCE_DEFAULT = os.environ.get("XHAND_SOURCE", "press_button_4_times")

# 目标目录默认落在 workspace 的 datasets/ 下；source env.sh 后 $ZEVA_WORK 已就位。
_TARGET_DEFAULT = os.path.join(os.environ.get("ZEVA_WORK", "."), "datasets/press_button_4_times_merged_filtered")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", default=SOURCE_DEFAULT, help="源 LeRobot 根目录（只读）")
    p.add_argument(
        "--target",
        default=os.environ.get("XHAND_DATA_ROOT", _TARGET_DEFAULT),
        help="目标根目录；默认取 $XHAND_DATA_ROOT，未设则用 $ZEVA_WORK/datasets/...",
    )
    p.add_argument("--category", default="xhand", help="任务大类，写入 task_category")
    p.add_argument("--task-name", default="PressButton4Times", help="任务名，写入 task_cluster")
    p.add_argument("--variant", default="lerobot_v21", help="三级子目录名，仅用于凑齐布局约定")
    p.add_argument(
        "--link",
        choices=("symlink", "hardlink", "copy"),
        default="symlink",
        help="data/videos 的引用方式（默认 symlink，不占额外空间）",
    )
    p.add_argument("--min-length", type=int, default=0, help="过滤掉短于该帧数的 episode（0=全保留）")
    p.add_argument("--max-episodes", type=int, default=0, help="只保留前 N 个 episode（0=全保留）")
    p.add_argument("--force", action="store_true", help="目标已存在时先删除再重建")
    return p.parse_args()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records))


def link_tree(src: Path, dst: Path, mode: str) -> int:
    """把 src 目录以软链/硬链/复制方式落到 dst，返回处理的文件数。"""
    count = 0
    if mode == "symlink":
        # 整目录一个软链，最省事
        dst.symlink_to(src.resolve(), target_is_directory=True)
        return sum(1 for _ in src.rglob("*") if _.is_file())
    dst.mkdir(parents=True, exist_ok=True)
    for item in sorted(src.rglob("*")):
        rel = item.relative_to(src)
        out = dst / rel
        if item.is_dir():
            out.mkdir(parents=True, exist_ok=True)
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        if mode == "hardlink":
            try:
                os.link(item, out)
            except OSError:
                shutil.copy2(item, out)
        else:
            shutil.copy2(item, out)
        count += 1
    return count


def main() -> int:
    args = parse_args()
    src = Path(args.source).resolve()
    tgt = Path(args.target).resolve()

    if not (src / "meta" / "info.json").is_file():
        print(f"错误：{src} 不像 LeRobot 数据集（缺 meta/info.json）", file=sys.stderr)
        return 2

    info = json.loads((src / "meta" / "info.json").read_text())
    if info.get("codebase_version") != "v2.1":
        print(f"警告：源 codebase_version={info.get('codebase_version')}，预期 v2.1", file=sys.stderr)

    root = tgt / args.category / args.task_name / args.variant / "lerobot"
    if root.exists():
        if not args.force:
            print(f"错误：{root} 已存在。加 --force 覆盖，或换个 --target。", file=sys.stderr)
            return 2
        shutil.rmtree(root)
    root.mkdir(parents=True)

    # --- meta ---
    # 小文件复制，让转换结果自描述；
    # 大文件（episodes_stats.jsonl 可达数 GB）改为软链，避免无谓占空间——
    # Cosmos 的加载器只读 info.json / episodes.jsonl / parquet，不读这两个。
    meta_out = root / "meta"
    meta_out.mkdir()
    COPY_LIMIT = 8 << 20  # 超过 8MB 的 meta 文件一律软链，不复制
    for name in ("info.json", "episodes.jsonl", "tasks.jsonl", "stats.json", "episodes_stats.jsonl"):
        s = src / "meta" / name
        if not s.is_file():
            continue
        if s.stat().st_size <= COPY_LIMIT:
            shutil.copy2(s, meta_out / name)
        else:
            (meta_out / name).symlink_to(s.resolve())
            print(f"  注：{name} ({s.stat().st_size/1e6:.1f} MB) 以软链引用，未复制")

    # 从 stats.json 抽出精简的 action/state 统计（原始文件上百 MB），
    # 供加载器的 action_normalization 使用。
    s = src / "meta" / "stats.json"
    if s.is_file():
        raw = json.loads(s.read_text())
        keep = {
            feat: {k: v for k, v in stats.items() if isinstance(v, list)}
            for feat, stats in raw.items()
            if feat in ("action", "observation.state") and isinstance(stats, dict)
        }
        if keep:
            out = meta_out / "feature_stats_compact.json"
            out.write_text(json.dumps(keep, ensure_ascii=False) + "\n")
            print(f"  已生成精简统计 {out.name} ({out.stat().st_size/1e6:.1f} MB)")

    # --- episode 过滤 ---
    records = read_jsonl(meta_out / "episodes.jsonl")
    total_before = len(records)
    if args.min_length > 0:
        records = [r for r in records if int(r["length"]) > args.min_length]
    if args.max_episodes > 0:
        records = records[: args.max_episodes]
    if len(records) != total_before:
        write_jsonl(meta_out / "episodes.jsonl", records)
        # 同步裁剪 info.json 的统计，避免下游读到不一致的规模
        info["total_episodes"] = len(records)
        info["total_frames"] = int(sum(int(r["length"]) for r in records))
        info["total_videos"] = int(info["total_videos"] * len(records) / total_before) if total_before else 0
        (meta_out / "info.json").write_text(json.dumps(info, indent=2, ensure_ascii=False) + "\n")

    # --- data / videos：零拷贝引用 ---
    n_data = link_tree(src / "data", root / "data", args.link)
    n_video = link_tree(src / "videos", root / "videos", args.link)

    # --- 报告 ---
    kept = {int(r["episode_index"]) for r in records}
    cams = sorted(p.name for p in (src / "videos" / "chunk-000").iterdir() if p.is_dir())
    print(f"源        : {src}")
    print(f"目标      : {root}")
    print(f"引用方式  : {args.link}")
    print(f"category  : {args.category}")
    print(f"task_name : {args.task_name}")
    print(f"episodes  : {len(records)}/{total_before} 保留  (帧数 {sum(int(r['length']) for r in records)})")
    print(f"fps       : {info['fps']}   （加载器需按此设置，非 RoboCasa 的 20）")
    print(f"相机      : {', '.join(cams)}")
    print(f"data 文件 : {n_data} 个")
    print(f"视频文件  : {n_video} 个")
    # 只统计真实占用：软链本身按 0 计，避免把源数据的体积算进来
    real = sum(f.lstat().st_size for f in root.rglob("*") if not f.is_symlink() and f.is_file())
    linked = sum(1 for f in root.rglob("*") if f.is_symlink())
    print(f"磁盘占用  : {real/1e6:.2f} MB（另有 {linked} 个软链引用源数据，不占空间）")

    # --- 自检：加载器会做的路径推导，这里先验证一遍 ---
    missing = []
    for r in records:
        ep = int(r["episode_index"])
        pq = root / f"data/chunk-{ep // 1000:03d}/episode_{ep:06d}.parquet"
        if not pq.is_file():
            missing.append(str(pq))
        for key in cams:
            v = root / f"videos/chunk-{ep // 1000:03d}/{key}/episode_{ep:06d}.mp4"
            if not v.is_file():
                missing.append(str(v))
    if missing:
        print(f"\n自检失败：{len(missing)} 个文件缺失，前 5 个：", file=sys.stderr)
        for m in missing[:5]:
            print(f"  {m}", file=sys.stderr)
        return 1
    print("\n自检通过：所有 episode 的 parquet 与视频均可解析。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
