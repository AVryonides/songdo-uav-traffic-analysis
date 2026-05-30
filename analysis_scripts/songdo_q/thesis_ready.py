"""Create title-free thesis-ready diagram exports for Q outputs.

This module intentionally does not modify the analysis figures in-place.  It
creates a separate ``thesis_ready_diagrams`` folder inside one Q run directory
and writes title-free PNG copies with descriptive file names plus a manifest.
"""

from __future__ import annotations

import csv
import os
import re
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageOps


_SKIP_DIR_NAMES = {"thesis_ready_diagrams"}


def _safe_name(text: str) -> str:
    """Return a readable filesystem-safe name."""
    text = str(text).strip()
    text = text.replace(":", "")
    text = re.sub(r"[^A-Za-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text.lower() or "diagram"


def _descriptive_filename(relative_png: Path) -> str:
    """Build a descriptive Q filename from the source plot path."""
    parts = list(relative_png.with_suffix("").parts)
    safe_parts = [_safe_name(p) for p in parts if p not in ("", ".")]
    return "q_intersection_" + "__".join(safe_parts) + "__title_free.png"


def _category_title_search_height(source_relative: Path, height: int) -> int:
    """Return how far down to search for the visible title text."""
    rel = str(source_relative).lower()
    if rel.endswith("headway_vs_vehicle_order_in_queue.png"):
        # This is the multi-panel lane comparison figure. Its per-panel lane
        # labels are intentionally retained; the source plot is generated
        # without a global figure title.
        return 0
    if rel.endswith("composite_space_time_diagram_plot_all.png"):
        # The all-lanes composite is generated without a global title; keep
        # the per-panel lane titles visible.
        return 0
    if "traffic_light_capped_median_pairwise_plot" in rel:
        return min(int(0.18 * height), 180)
    if "headway_vs_vehicle_order_in_queue_by_lane" in rel:
        return min(int(0.13 * height), 150)
    if "headway_vs_vehicle_order_in_queue" in rel:
        return min(int(0.14 * height), 180)
    if "signal_timing" in rel or "phase_summary" in rel:
        return min(int(0.13 * height), 170)
    if "composite_space_time" in rel:
        return min(int(0.10 * height), 180)
    return min(int(0.12 * height), 170)


def _detect_top_title_boxes(img: Image.Image, source_relative: Path) -> list[tuple[int, int, int, int]]:
    """Detect likely title text boxes near the top of a Matplotlib figure.

    This masks title text without cropping the image.  It keeps the plot
    geometry intact, which is important for legends, annotations, and axes.
    """
    import numpy as np

    rgb = img.convert("RGB")
    arr = np.asarray(rgb)
    height, width = arr.shape[:2]
    scan_h = _category_title_search_height(source_relative, height)
    if scan_h <= 0:
        return []

    top = arr[:scan_h]
    dark = (top[:, :, 0] < 115) & (top[:, :, 1] < 115) & (top[:, :, 2] < 115)
    x0 = int(width * 0.12)
    x1 = int(width * 0.88)
    centered_dark = dark[:, x0:x1]
    row_counts = centered_dark.sum(axis=1)
    threshold = max(14, int(width * 0.010))

    bands: list[tuple[int, int]] = []
    in_band = False
    start = 0
    for y, count in enumerate(row_counts):
        if count > threshold and not in_band:
            start = y
            in_band = True
        elif (count <= threshold or y == len(row_counts) - 1) and in_band:
            end = y - 1 if count <= threshold else y
            if end - start >= 2:
                bands.append((start, end))
            in_band = False

    boxes: list[tuple[int, int, int, int]] = []
    for start, end in bands:
        if start > max(120, int(height * 0.08)):
            break

        band_dark = dark[start:end + 1]
        ys, xs = np.where(band_dark)
        if len(xs) == 0:
            continue

        left = max(0, int(xs.min()) - 14)
        right = min(width, int(xs.max()) + 15)
        top_y = max(0, start - 6)
        bottom_y = min(height, end + 8)
        center_x = 0.5 * (left + right)
        box_width = right - left
        box_height = bottom_y - top_y

        if not (0.25 * width <= center_x <= 0.75 * width):
            continue
        if box_width < 0.10 * width:
            continue
        if box_width > 0.75 * width and box_height < 22:
            continue

        boxes.append((left, top_y, right, bottom_y))
        if len(boxes) >= 2:
            break

    return boxes


def _make_title_free_png(source: Path, destination: Path, source_relative: Path) -> tuple[int, int, int]:
    """Mask the visible title text and save a clean PNG copy."""
    with Image.open(source) as img:
        img = ImageOps.exif_transpose(img)
        width, height = img.size
        canvas = img.convert("RGB")
        boxes = _detect_top_title_boxes(canvas, source_relative)
        for left, top, right, bottom in boxes:
            patch = Image.new("RGB", (right - left, bottom - top), "white")
            canvas.paste(patch, (left, top))
        destination.parent.mkdir(parents=True, exist_ok=True)
        canvas.save(destination, "PNG", optimize=True)
        return width, height, len(boxes)


def _iter_pngs(run_dir: Path) -> Iterable[Path]:
    for png_path in sorted(run_dir.rglob("*.png")):
        if any(part in _SKIP_DIR_NAMES for part in png_path.relative_to(run_dir).parts):
            continue
        yield png_path


def create_q_thesis_ready_diagrams(run_dir: str | os.PathLike) -> Path:
    """Create title-free thesis-ready diagram exports for one Q run folder."""
    run_path = Path(run_dir).resolve()
    if not run_path.exists():
        raise FileNotFoundError(f"Q run folder not found: {run_path}")

    out_dir = run_path / "general" / "thesis_ready_diagrams"
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    for src in _iter_pngs(run_path):
        rel = src.relative_to(run_path)
        # Keep a readable category structure, but make every file self-descriptive.
        category = rel.parts[0] if len(rel.parts) > 1 else "root"
        dst = out_dir / _safe_name(category) / _descriptive_filename(rel)
        width, height, title_boxes_removed = _make_title_free_png(src, dst, rel)
        manifest_rows.append({
            "source_file": str(rel),
            "thesis_ready_file": str(dst.relative_to(out_dir)),
            "original_width_px": width,
            "original_height_px": height,
            "title_boxes_removed": title_boxes_removed,
        })

    manifest_path = out_dir / "thesis_ready_diagram_manifest.csv"
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "source_file",
                "thesis_ready_file",
                "original_width_px",
                "original_height_px",
                "title_boxes_removed",
            ],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    return out_dir


def _latest_q_run(outputs_dir: str | os.PathLike = "outputs_Q") -> Path:
    candidates = sorted(Path(outputs_dir).glob("vehicle_*_2022-10-04_Q_PM5"))
    if not candidates:
        raise FileNotFoundError(f"No Q run folders found in {outputs_dir}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        default=None,
        help="Q run directory. Defaults to the most recently modified outputs_Q/vehicle_*_2022-10-04_Q_PM5 folder.",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir) if args.run_dir else _latest_q_run()
    out_dir = create_q_thesis_ready_diagrams(run_dir)
    print(f"Thesis-ready Q diagrams written to: {out_dir}")


if __name__ == "__main__":
    main()
