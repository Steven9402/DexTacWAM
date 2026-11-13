#!/usr/bin/env python3
"""Day 05 标准答案：T-Rex LeRobot v3.0 → DexTacWAM LeRobot v2.x。

本脚本只做数据转换、落盘和读回校验；不做归一化、统计、缓存或训练。

最重要的维度合同（parquet 中一行就是一帧）：
    T                  当前 episode 的帧数；30 fps 时，T / 30 秒。
    state[t]           (58,) float32，当前帧的机器人关节状态。
    action[t]          (58,) float32，当前帧的机器人关节目标。
    force[t]           (60,) float32，当前帧双手五指的六维力/力矩观测。
    actions[t]         (118,) float32 = [force[t] (60) | action[t] (58)]。
    RGB PNG 解码后      (192, 256, 3) uint8 = [高, 宽, RGB 通道]。
    tactile[t]         (2, 5, 192, 256) uint8
                       = [左右手, 五指, 灰度图高, 灰度图宽]。

58 维 state/action 的物理顺序：
    [0:7]   左臂 7 个关节；臂关节通常以弧度表示。
    [7:29]  左手 22 个关节；Sharpa 手的关节角/归一化行程，沿用源数据单位。
    [29:36] 右臂 7 个关节。
    [36:58] 右手 22 个关节。
转换器不改变这些数值的单位，也不把 state 错当成 action。

60 维 force 的物理顺序：
    [0:30]  左手五指，每指连续 6 维 (fx, fy, fz, mx, my, mz)。
    [30:60] 右手五指，顺序相同。
    五指顺序均为 thumb/index/middle/ring/pinky。
    f* 是力，m* 是力矩；本脚本原样保留源数据单位，不猜测或换算单位。

单集转换：
    conda activate trex-quickstart
    python scripts/convert_trex_to_dexvtam.py \
      --src ~/.cache/trex_quickstart \
      --quickstart-src ../T-Rex/dataset_quickstart/src \
      --out data/datasets_lerobot/trex_wipe \
      --episodes ID0,ID1,ID2,ID3,ID4,ID5,ID6,ID7 \
      --max-episodes 1

单集检查通过后去掉 --max-episodes。已存在且校验通过的 parquet 会复用；
只有显式传 --overwrite 才会重转换并覆盖。另开进程全量读回：
    python scripts/convert_trex_to_dexvtam.py \
      --out data/datasets_lerobot/trex_wipe --validate-only
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

FPS = 30
STATE_DIM = 58
JOINT_ACTION_DIM = 58
FORCE_DIM = 60
TARGET_ACTION_DIM = FORCE_DIM + JOINT_ACTION_DIM  # 118

RGB_SOURCE_SHAPE = (360, 640, 3)  # [H, W, C]，16:9 RGB。
TACTILE_SOURCE_SHAPE = (240, 320)  # [H, W]，4:3 单通道灰度。
TARGET_HEIGHT = 192
TARGET_WIDTH = 256
RGB_TARGET_SHAPE = (TARGET_HEIGHT, TARGET_WIDTH, 3)
TACTILE_TARGET_SHAPE = (2, 5, TARGET_HEIGHT, TARGET_WIDTH)

SIDES = ("left", "right")  # 第 0 维：左手在前、右手在后。
FINGERS = ("thumb", "index", "middle", "ring", "pinky")  # 第 1 维。
RGB_KEYS = {
    "head_img": "observation.images.head_left",
    "left_wrist_img": "observation.images.left_wrist",
    "right_wrist_img": "observation.images.right_wrist",
}
RGB_RESAMPLE = Image.Resampling.LANCZOS
TACTILE_RESAMPLE = Image.Resampling.BILINEAR

IMAGE_STRUCT = pa.struct([("bytes", pa.binary()), ("path", pa.string())])
TACTILE_TYPE = pa.list_(pa.list_(pa.list_(pa.list_(pa.uint8()))))
PARQUET_SCHEMA = pa.schema(
    [
        ("head_img", IMAGE_STRUCT),
        ("left_wrist_img", IMAGE_STRUCT),
        ("right_wrist_img", IMAGE_STRUCT),
        ("tactile", TACTILE_TYPE),
        ("state", pa.list_(pa.float32())),
        ("actions", pa.list_(pa.float32())),
        ("frame_index", pa.int64()),
        ("episode_index", pa.int64()),
    ]
)


def load_episode_row(src_root: str | Path, source_episode_index: int) -> pd.Series:
    """返回源 episode 唯一的元数据行，保留共享 mp4 的定位字段。

    ep_row 的关键字段：
        episode_index/length/caption/object：源编号、帧数、任务文本、物体。
        data/chunk_index + data/file_index：数值 parquet 的位置。
        videos/<key>/chunk_index + file_index + from_timestamp：
            每路视频的位置和本 episode 在共享 mp4 中的起始秒数。
    """
    src_root = Path(src_root).expanduser()
    files = sorted((src_root / "meta" / "episodes").glob("**/*.parquet"))
    if not files:
        raise FileNotFoundError(f"未找到源 episode 元数据：{src_root}/meta/episodes")

    matches = []
    for path in files:
        metadata = pd.read_parquet(path)
        selected = metadata.loc[metadata["episode_index"] == source_episode_index]
        if not selected.empty:
            matches.append(selected)
    if len(matches) != 1 or len(matches[0]) != 1:
        raise ValueError(f"源 episode {source_episode_index} 必须恰好出现一次")

    ep_row = matches[0].iloc[0]
    if int(ep_row["length"]) <= 0:
        raise ValueError(f"源 episode {source_episode_index} 的 length 非正数")
    if not isinstance(ep_row["caption"], str) or not ep_row["caption"].strip():
        raise ValueError(f"源 episode {source_episode_index} 缺少有效 caption")
    return ep_row


def _load_source_frame_index(src_root: Path, ep_row: pd.Series) -> np.ndarray:
    """按 data_path 精确读取一集的 frame_index，并按源帧号排序。"""
    info = json.loads((src_root / "meta" / "info.json").read_text(encoding="utf-8"))
    relative_path = info["data_path"].format(
        chunk_index=int(ep_row["data/chunk_index"]),
        file_index=int(ep_row["data/file_index"]),
    )
    columns = ["episode_index", "frame_index"]
    data = pd.read_parquet(src_root / relative_path, columns=columns)
    source_ep = int(ep_row["episode_index"])
    data = data.loc[data["episode_index"] == source_ep].sort_values("frame_index")
    return data["frame_index"].to_numpy(dtype=np.int64)


def load_numeric(
    src_root: str | Path, ep_row: pd.Series
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """读取 raw 数值，返回 (state, actions, frame_index)。

    返回形状和物理含义：
        state       (T,58) float32：每帧 [左臂7|左手22|右臂7|右手22]。
        actions     (T,118) float32：每帧 [双手六维力/力矩60|关节目标58]。
        frame_index (T,) int64：排序后的源帧号，必须为 0..T-1。

    这里刻意不归一化。q01/q99 归一化是 DexTacWAM Dataset 后续的职责。
    """
    from trex_dataset_quickstart.decode import (
        load_episode_action,
        load_episode_f6,
        load_episode_state,
    )

    src_root = Path(src_root).expanduser()
    T = int(ep_row["length"])
    source_ep = int(ep_row["episode_index"])
    state = load_episode_state(src_root, source_ep, T).astype(np.float32)
    joint_action = load_episode_action(src_root, source_ep, T).astype(np.float32)
    force = load_episode_f6(src_root, source_ep, T).astype(np.float32)
    frame_index = _load_source_frame_index(src_root, ep_row)

    if state.shape != (T, STATE_DIM):
        raise ValueError(f"state shape {state.shape} != {(T, STATE_DIM)}")
    if joint_action.shape != (T, JOINT_ACTION_DIM):
        raise ValueError(f"action shape {joint_action.shape} != {(T, JOINT_ACTION_DIM)}")
    if force.shape != (T, FORCE_DIM):
        raise ValueError(f"force shape {force.shape} != {(T, FORCE_DIM)}")
    for name, array in (("state", state), ("action", joint_action), ("force", force)):
        if not np.isfinite(array).all():
            raise ValueError(f"{name} 含 NaN/Inf；停止转换，不用 0 掩盖异常")

    # axis=1 是拼每帧的特征维，不是拼时间维；顺序必须是力在前、关节目标在后。
    actions = np.concatenate([force, joint_action], axis=1).astype(np.float32)
    if actions.shape != (T, TARGET_ACTION_DIM):
        raise ValueError(f"actions shape {actions.shape} != {(T, TARGET_ACTION_DIM)}")
    if not np.array_equal(frame_index, np.arange(T, dtype=np.int64)):
        raise ValueError("源 frame_index 排序后不是连续的 0..T-1")
    return state, actions, frame_index


def rgb_frames_to_png_cells(frames: np.ndarray) -> list[dict[str, bytes | str]]:
    """(T,360,640,3) uint8 → T 个可嵌入 parquet 的 PNG 字典。

    每帧先从 16:9 中心裁成 4:3：(360,640,3) → (360,480,3)，
    再缩放为 (192,256,3)。这样不会把机器人横向压扁。
    输出 bytes 是无损 PNG；path 留空，因为图像内容直接存在 bytes 中。
    """
    if frames.ndim != 4 or frames.shape[1:] != RGB_SOURCE_SHAPE:
        raise ValueError(f"RGB frames shape {frames.shape}，期望 (T,{RGB_SOURCE_SHAPE})")
    if frames.dtype != np.uint8:
        raise ValueError(f"RGB frames dtype {frames.dtype} != uint8")

    source_h, source_w = frames.shape[1:3]
    crop_w = source_h * TARGET_WIDTH // TARGET_HEIGHT  # 360*256/192 = 480。
    x0 = (source_w - crop_w) // 2  # (640-480)//2 = 80，左右各裁 80 列。
    cells: list[dict[str, bytes | str]] = []
    for frame in frames:
        cropped = frame[:, x0 : x0 + crop_w, :]
        resized = Image.fromarray(cropped, mode="RGB").resize(
            (TARGET_WIDTH, TARGET_HEIGHT), RGB_RESAMPLE  # PIL 参数是 (W,H)。
        )
        rgb = np.asarray(resized, dtype=np.uint8)
        if rgb.shape != RGB_TARGET_SHAPE:
            raise ValueError(f"缩放后 RGB shape {rgb.shape} != {RGB_TARGET_SHAPE}")
        buffer = io.BytesIO()
        resized.save(buffer, format="PNG")
        cells.append({"bytes": buffer.getvalue(), "path": ""})
    return cells


def convert_rgb_key(
    src_root: str | Path, ep_row: pd.Series, video_key: str
) -> list[dict[str, bytes | str]]:
    """解码一路共享 mp4，返回长度为 T 的 PNG 字典列表。"""
    from trex_dataset_quickstart.decode import decode_key

    T = int(ep_row["length"])
    frames = decode_key(src_root, video_key, ep_row, T, gray=False)
    try:
        return rgb_frames_to_png_cells(frames)
    finally:
        del frames  # 一次只保留一路 RGB 大数组，避免内存随相机数增长。


def convert_tactile(src_root: str | Path, ep_row: pd.Series) -> np.ndarray:
    """10 路 raw 灰度视频 → (T,2,5,192,256) uint8。

    五个轴依次是：
        T：时间；2：left/right；5：thumb/index/middle/ring/pinky；
        192：灰度图高度；256：灰度图宽度。
    只读取 raw，不读取 deform。源和目标都是 4:3，直接缩放不会变形。
    """
    from trex_dataset_quickstart.decode import decode_key
    from trex_dataset_quickstart.schema import tactile_video_key

    T = int(ep_row["length"])
    out = np.empty((T, *TACTILE_TARGET_SHAPE), dtype=np.uint8)
    for side_idx, side in enumerate(SIDES):
        for finger_idx, finger in enumerate(FINGERS):
            key = tactile_video_key(side, "raw", finger)
            frames = decode_key(src_root, key, ep_row, T, gray=True)
            if frames.shape != (T, *TACTILE_SOURCE_SHAPE):
                raise ValueError(f"{key} shape {frames.shape} != {(T, *TACTILE_SOURCE_SHAPE)}")
            if frames.dtype != np.uint8:
                raise ValueError(f"{key} dtype {frames.dtype} != uint8")
            for t in range(T):
                resized = Image.fromarray(frames[t], mode="L").resize(
                    (TARGET_WIDTH, TARGET_HEIGHT), TACTILE_RESAMPLE
                )
                out[t, side_idx, finger_idx] = np.asarray(resized, dtype=np.uint8)
            del frames
    return out


def convert_episode(
    output_path: str | Path,
    new_episode_index: int,
    head_cells: list[dict[str, bytes | str]],
    left_cells: list[dict[str, bytes | str]],
    right_cells: list[dict[str, bytes | str]],
    tactile: np.ndarray,
    state: np.ndarray,
    actions: np.ndarray,
    frame_index: np.ndarray,
    overwrite: bool = False,
) -> None:
    """把一集写成每帧一行的 parquet；显式 schema 防止 dtype 被推断改变。

    整集数组在写入前的形状：
        head/left/right_cells：长度 T；每项解码后为 (192,256,3) uint8。
        tactile：(T,2,5,192,256) uint8；第 t 行只写 tactile[t]。
        state：(T,58) float32；第 t 行只写 state[t]。
        actions：(T,118) float32；第 t 行只写 actions[t]。
        frame_index：(T,) int64；episode_index 是标量，在每行重复。
    """
    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"拒绝覆盖 {output_path}；如确需覆盖请传 --overwrite")

    T = len(frame_index)
    if tactile.shape != (T, *TACTILE_TARGET_SHAPE):
        raise ValueError(f"tactile shape {tactile.shape} != {(T, *TACTILE_TARGET_SHAPE)}")
    if state.shape != (T, STATE_DIM) or state.dtype != np.float32:
        raise ValueError("state 必须是 (T,58) float32")
    if actions.shape != (T, TARGET_ACTION_DIM) or actions.dtype != np.float32:
        raise ValueError("actions 必须是 (T,118) float32")
    if not all(len(cells) == T for cells in (head_cells, left_cells, right_cells)):
        raise ValueError("三路 RGB 的 PNG cell 数都必须等于 T")

    rows = []
    for t in range(T):
        rows.append(
            {
                "head_img": head_cells[t],
                "left_wrist_img": left_cells[t],
                "right_wrist_img": right_cells[t],
                # 四维 [手,指,H,W] → 四层 list；schema 把最内层固定为 uint8。
                "tactile": tactile[t].tolist(),
                "state": state[t],
                "actions": actions[t],
                "frame_index": int(frame_index[t]),
                "episode_index": int(new_episode_index),
            }
        )
    dataframe = pd.DataFrame(rows)
    if len(dataframe) != T:
        raise ValueError("parquet 行数与 T 不一致")
    if list(dataframe["episode_index"].unique()) != [new_episode_index]:
        raise ValueError("一集内出现多个 episode_index")
    table = pa.Table.from_pandas(dataframe, schema=PARQUET_SCHEMA, preserve_index=False)

    # 临时文件写完再原子替换，异常时不会留下半个正式 parquet。
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=output_path.parent, prefix=output_path.name + ".", suffix=".tmp", delete=False
        ) as temporary:
            temporary_path = Path(temporary.name)
        pq.write_table(table, temporary_path)
        if output_path.exists() and not overwrite:
            raise FileExistsError(f"拒绝覆盖 {output_path}")
        temporary_path.replace(output_path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _deep_stack(value) -> np.ndarray:
    """与 DexTacWAM 的读取方式一致：嵌套 object 数组 → dense ndarray。"""
    if isinstance(value, np.ndarray) and value.dtype == object:
        return np.stack([_deep_stack(item) for item in value])
    return np.asarray(value)


def validate_episode(
    output_path: str | Path,
    expected_episode_index: int,
    expected_length: int,
    expected_state: np.ndarray | None = None,
    expected_actions: np.ndarray | None = None,
    expected_rgb_cells: dict[str, list[dict[str, bytes | str]]] | None = None,
    expected_tactile: np.ndarray | None = None,
) -> None:
    """只从磁盘读回，检查 schema、shape、dtype、编号、数值和图像内容。

    可选 expected_* 用于转换后的逐元素对账；--validate-only 另开进程时，
    不依赖内存中的源数组，仍检查完整存储合同和首/中/尾帧可解码性。
    """
    output_path = Path(output_path)
    actual_schema = pq.read_schema(output_path)
    if not actual_schema.equals(PARQUET_SCHEMA, check_metadata=False):
        raise ValueError(f"parquet schema 不符合合同：\n{actual_schema}")

    dataframe = pd.read_parquet(output_path)
    T = len(dataframe)
    if T != expected_length or T <= 0:
        raise ValueError(f"行数 {T} != expected_length {expected_length}")
    if list(dataframe.columns) != PARQUET_SCHEMA.names:
        raise ValueError(f"列名或列顺序错误：{list(dataframe.columns)}")
    if not np.array_equal(dataframe["frame_index"].to_numpy(), np.arange(T)):
        raise ValueError("frame_index 不是 0..T-1")
    episode_ids = dataframe["episode_index"].to_numpy()
    if not np.all(episode_ids == expected_episode_index):
        raise ValueError("episode_index 与文件名/新编号不一致")

    state = np.stack(dataframe["state"].to_numpy()).astype(np.float32, copy=False)
    actions = np.stack(dataframe["actions"].to_numpy()).astype(np.float32, copy=False)
    if state.shape != (T, STATE_DIM):
        raise ValueError(f"读回 state shape {state.shape} != {(T, STATE_DIM)}")
    if actions.shape != (T, TARGET_ACTION_DIM):
        raise ValueError(f"读回 actions shape {actions.shape} != {(T, TARGET_ACTION_DIM)}")
    if not np.isfinite(state).all() or not np.isfinite(actions).all():
        raise ValueError("读回 state/actions 含 NaN/Inf")
    if expected_state is not None and not np.array_equal(state, expected_state):
        raise ValueError("读回 state 与源 observation.state 的 float32 值不一致")
    if expected_actions is not None and not np.array_equal(actions, expected_actions):
        raise ValueError("读回 actions 与 [源 force60 | 源 action58] 不一致")

    sample_indexes = sorted({0, T // 2, T - 1})
    for camera in RGB_KEYS:
        for t in range(T):  # 每帧都验证 PNG 可解码，首/中/尾再逐像素对账。
            cell = dataframe.iloc[t][camera]
            if cell["path"] != "":
                raise ValueError(f"{camera}[{t}].path 应为空字符串")
            with Image.open(io.BytesIO(cell["bytes"])) as image:
                rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
            if rgb.shape != RGB_TARGET_SHAPE or rgb.dtype != np.uint8:
                raise ValueError(f"{camera}[{t}] 不是 (192,256,3) uint8")
            if expected_rgb_cells is not None and t in sample_indexes:
                with Image.open(io.BytesIO(expected_rgb_cells[camera][t]["bytes"])) as image:
                    expected_rgb = np.asarray(image.convert("RGB"), dtype=np.uint8)
                if not np.array_equal(rgb, expected_rgb):
                    raise ValueError(f"{camera}[{t}] PNG 像素与裁剪缩放结果不一致")

    for t in sample_indexes:
        tactile = _deep_stack(dataframe.iloc[t]["tactile"])
        if tactile.shape != TACTILE_TARGET_SHAPE or tactile.dtype != np.uint8:
            raise ValueError(f"tactile[{t}] 不是 (2,5,192,256) uint8")
        if expected_tactile is not None and not np.array_equal(tactile, expected_tactile[t]):
            raise ValueError(f"tactile[{t}] 与 raw 灰度缩放结果不一致")


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent, prefix=path.name + ".",
            suffix=".tmp", delete=False
        ) as temporary:
            temporary.write(content)
            temporary_path = Path(temporary.name)
        temporary_path.replace(path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def write_meta(out_root: str | Path, converted: list[dict]) -> None:
    """所有本批次 episode 成功后，写 info/tasks/episodes 和追溯 manifest。

    converted 每项：
        new_episode_index：目标连续编号 0..N-1，同时决定 parquet 路径。
        source_episode_index：T-Rex 原编号，仅用于追溯。
        length=T、caption、object：源 episode 的帧数、任务文本、物体。
    """
    out_root = Path(out_root)
    if not converted:
        raise ValueError("没有已转换 episode，拒绝写空 meta")
    converted = sorted(converted, key=lambda item: item["new_episode_index"])
    new_ids = [item["new_episode_index"] for item in converted]
    if new_ids != list(range(len(converted))):
        raise ValueError(f"目标 episode 编号必须连续为 0..N-1，实际为 {new_ids}")

    captions = list(dict.fromkeys(item["caption"] for item in converted))
    tasks = [{"task_index": index, "task": caption} for index, caption in enumerate(captions)]
    episodes = [
        {
            "episode_index": item["new_episode_index"],
            "tasks": [item["caption"]],  # 是字符串列表，不是 task_index。
            "length": item["length"],
        }
        for item in converted
    ]
    chunks_size = 1000
    info = {
        "codebase_version": "v2.1",
        "fps": FPS,
        "total_episodes": len(converted),
        "total_frames": sum(item["length"] for item in converted),
        "total_tasks": len(captions),
        "total_chunks": max(new_ids) // chunks_size + 1,
        "chunks_size": chunks_size,
        "data_path": "data/chunk-{chunk_index:03d}/episode_{episode_index:06d}.parquet",
    }
    meta = out_root / "meta"
    _atomic_write_text(meta / "info.json", json.dumps(info, ensure_ascii=False, indent=2) + "\n")
    _atomic_write_text(
        meta / "tasks.jsonl",
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in tasks),
    )
    _atomic_write_text(
        meta / "episodes.jsonl",
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in episodes),
    )
    _atomic_write_text(
        meta / "conversion_manifest.jsonl",
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in converted),
    )


def _episode_output_path(out_root: Path, episode_index: int, chunks_size: int = 1000) -> Path:
    chunk = episode_index // chunks_size
    return out_root / "data" / f"chunk-{chunk:03d}" / f"episode_{episode_index:06d}.parquet"


def _load_existing_manifest(out_root: Path) -> dict[int, dict]:
    path = out_root / "meta" / "conversion_manifest.jsonl"
    if not path.exists():
        return {}
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    return {int(record["new_episode_index"]): record for record in records}


def _parse_episode_ids(text: str) -> list[int]:
    try:
        episode_ids = [int(item.strip()) for item in text.split(",") if item.strip()]
    except ValueError as exc:
        raise ValueError("--episodes 必须是逗号分隔的整数") from exc
    if not episode_ids or len(episode_ids) != len(set(episode_ids)):
        raise ValueError("--episodes 不能为空或含重复编号")
    return episode_ids


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, help="T-Rex 本地数据集根目录")
    parser.add_argument("--quickstart-src", type=Path, help="trex_dataset_quickstart 的 src 目录")
    parser.add_argument("--out", type=Path, required=True, help="目标 v2.x 数据集根目录")
    parser.add_argument("--episodes", help="逗号分隔的 T-Rex 原编号；顺序决定新编号 0..N-1")
    parser.add_argument("--max-episodes", type=int, help="只处理前 N 集，用于单集冒烟")
    parser.add_argument("--overwrite", action="store_true", help="显式允许重转换并覆盖旧 parquet")
    parser.add_argument("--validate-only", action="store_true", help="另开进程，仅从磁盘读回校验")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    out_root = args.out.expanduser()
    if args.max_episodes is not None and args.max_episodes <= 0:
        raise ValueError("--max-episodes 必须为正整数")

    if args.validate_only:
        episodes_path = out_root / "meta" / "episodes.jsonl"
        episodes = [json.loads(line) for line in episodes_path.read_text(encoding="utf-8").splitlines() if line]
        if args.max_episodes is not None:
            episodes = episodes[: args.max_episodes]
        for episode in episodes:
            path = _episode_output_path(out_root, int(episode["episode_index"]))
            validate_episode(path, int(episode["episode_index"]), int(episode["length"]))
            print(f"PASS: {path}")
        print(f"独立读回完成：{len(episodes)} 集")
        return

    if args.src is None or args.quickstart_src is None or not args.episodes:
        raise ValueError("转换模式必须提供 --src、--quickstart-src 和 --episodes")
    src_root = args.src.expanduser().resolve()
    quickstart_src = args.quickstart_src.expanduser().resolve()
    sys.path.insert(0, str(quickstart_src))  # 必须在调用 trex_dataset_quickstart 前设置。

    source_ids = _parse_episode_ids(args.episodes)
    if args.max_episodes is not None:
        source_ids = source_ids[: args.max_episodes]
    existing_manifest = _load_existing_manifest(out_root)
    converted: list[dict] = []

    for new_episode_index, source_episode_index in enumerate(source_ids):
        ep_row = load_episode_row(src_root, source_episode_index)
        T = int(ep_row["length"])
        output_path = _episode_output_path(out_root, new_episode_index)
        record = {
            "new_episode_index": new_episode_index,
            "source_episode_index": source_episode_index,
            "object": str(ep_row.get("object", "")),
            "length": T,
            "caption": ep_row["caption"],
        }

        if output_path.exists() and not args.overwrite:
            previous = existing_manifest.get(new_episode_index)
            if previous is not None and int(previous["source_episode_index"]) != source_episode_index:
                raise ValueError(
                    f"目标 episode {new_episode_index} 已对应源 {previous['source_episode_index']}，"
                    f"不能静默改对应源 {source_episode_index}；请检查清单或显式 --overwrite"
                )
            validate_episode(output_path, new_episode_index, T)
            converted.append(record)
            print(f"复用已校验文件：{output_path}")
            continue

        state, actions, frame_index = load_numeric(src_root, ep_row)
        rgb_cells = {
            output_key: convert_rgb_key(src_root, ep_row, source_key)
            for output_key, source_key in RGB_KEYS.items()
        }
        tactile = convert_tactile(src_root, ep_row)
        convert_episode(
            output_path=output_path,
            new_episode_index=new_episode_index,
            head_cells=rgb_cells["head_img"],
            left_cells=rgb_cells["left_wrist_img"],
            right_cells=rgb_cells["right_wrist_img"],
            tactile=tactile,
            state=state,
            actions=actions,
            frame_index=frame_index,
            overwrite=args.overwrite,
        )
        validate_episode(
            output_path=output_path,
            expected_episode_index=new_episode_index,
            expected_length=T,
            expected_state=state,
            expected_actions=actions,
            expected_rgb_cells=rgb_cells,
            expected_tactile=tactile,
        )
        converted.append(record)
        print(f"PASS: source={source_episode_index} → new={new_episode_index}, T={T}, {output_path}")
        del state, actions, frame_index, rgb_cells, tactile

    write_meta(out_root, converted)
    print(f"完成：{len(converted)} 集，{sum(item['length'] for item in converted)} 帧 → {out_root}")


if __name__ == "__main__":
    main()
