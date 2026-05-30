# config.py
import os
import re
import argparse
import matplotlib.pyplot as plt
import tool_songdo as tool
from tool_songdo_R import Master

VERBOSE = False  # True = debug spam, False = clean console
INTERSECTION_LABEL = "R"

def log(msg: str):
    print(msg)

def dbg(msg: str):
    if VERBOSE:
        print(msg)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", required=True, help="Trajectory CSV, e.g. 2022-10-04_R_PM5.csv")
    ap.add_argument("--seg", required=False, help="Segmentation CSV, e.g. R.csv. If omitted, inferred from traj filename.")
    ap.add_argument("--seg_dir", default=".", help="Directory where segmentation CSVs live (default: current dir)")
    ap.add_argument("--out", default="outputs_R", help="Output parent directory")
    ap.add_argument("--grouping-mode", default="rules", choices=["rules","auto","manual"],
                    help="How to group movements: 'rules' uses embedded rules, 'auto' infers approaches, 'manual' loads mapping file")
    ap.add_argument("--manual-mapping-file", required=False,
                    help="Path to JSON file with explicit movement rules when using --grouping-mode manual")
    ap.add_argument("--export-sections", required=False,
                    help="If provided, path to write section centroids CSV for inspection")
    return ap.parse_args()

def _save_fig_formats(fig, base_path, dpi=200):
    """Save a figure in PNG, SVG, and PDF formats."""
    fig.savefig(base_path, dpi=dpi)
    base_no_ext, _ = os.path.splitext(base_path)
    fig.savefig(base_no_ext + ".svg")
    fig.savefig(base_no_ext + ".pdf")

def with_intersection_title(title, intersection_label: str = INTERSECTION_LABEL):
    """Prefix visible plot titles with the intersection id once."""
    if title is None:
        return title
    text = str(title)
    if not text.strip():
        return text

    label = str(intersection_label).strip().upper()
    first_line = text.lstrip().splitlines()[0].strip()
    already_prefixed = (
        first_line == label
        or first_line.startswith(f"{label} ")
        or first_line.startswith(f"{label}-")
        or first_line.startswith(f"{label} -")
        or first_line.startswith(f"{label}—")
        or first_line.startswith(f"{label} —")
        or first_line.startswith(f"{label}:")
        or first_line.startswith(f"{label} Intersection")
    )
    if already_prefixed:
        return text
    return f"{label} - {text}"

# Thesis-quality font sizes for matplotlib
_THESIS_FONT = {
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 11,
    "figure.titlesize": 18,
}

def infer_intersection_id_from_filename(path: str) -> str:
    base = os.path.basename(path)
    parts = base.split("_")
    for p in parts:
        if len(p) == 1 and p.isalpha():
            return p.upper()
    m = re.search(r"_([A-Za-z])_", base)
    return m.group(1).upper() if m else "UNKNOWN"


def patch_od_pairs():
    """No-op: R intersection uses arm-number OD pairs; no refinement needed."""
    pass


def patch_matplotlib():
    """
    Patch plt.show/tool.plt.show to no-ops and prefix visible plot titles.
    Returns (restore_fn, original_close) so the caller can undo it.
    """
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    _original_show = plt.show
    _original_close = plt.close
    _original_set_title = Axes.set_title
    _original_suptitle = Figure.suptitle

    def _no_show(*args, **kwargs):
        pass

    def _set_title_with_intersection(ax_self, label, *args, **kwargs):
        return _original_set_title(
            ax_self, with_intersection_title(label), *args, **kwargs
        )

    def _suptitle_with_intersection(fig_self, t, *args, **kwargs):
        return _original_suptitle(
            fig_self, with_intersection_title(t), *args, **kwargs
        )

    plt.show = _no_show
    tool.plt.show = _no_show
    Axes.set_title = _set_title_with_intersection
    Figure.suptitle = _suptitle_with_intersection

    def restore():
        plt.show = _original_show
        tool.plt.show = _original_show
        Axes.set_title = _original_set_title
        Figure.suptitle = _original_suptitle

    return restore, _original_close
