"""Result figures for the cup stage (2026-11-17 recordings, learned selector).

Inputs are evaluation.json files written by `evaluate`, the replay runs with
--collect-candidates (for what any candidate could have reached), and the
reviewed annotations. Outputs:
- fig5_tradeoff.png: detection rate vs processing speed, pooled over the four
  leave-one-session-out folds (each session scored by a selector not trained on it)
- fig6_breakdown.png: per condition, where every visible frame ends up
- fig7_dark_example.png: what the detector sees in the held-out dark session
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from small_object_lab.research.evaluation import box_iou, load_annotations  # noqa: E402

spec = importlib.util.spec_from_file_location("progress", ROOT / "scripts/make_progress_figures.py")
progress = importlib.util.module_from_spec(spec)
spec.loader.exec_module(progress)
INK, INK2, MUTED, GRID, SURFACE = progress.INK, progress.INK2, progress.MUTED, progress.GRID, progress.SURFACE
BLUE, GRAY = "#2a78d6", "#8a8984"

METHOD_NAMES = {"baseline": "通常YOLO", "bicubic": "通常拡大（bicubic）", "fixed": "固定超解像",
                "rule": "動的倍率規則", "learned": "学習による選択"}
RESEARCH = ROOT / "outputs/research"
ANNOTATIONS = ROOT / "outputs/annotations"
# (label, session, run with all five methods, candidates run, annotations, role)
SESSIONS = [
    ("A・明るい", "20261117_021645_944529", "fold_cup_021645_944529", "cup_20261117_021645_944529", "cup_20261117_021645_944529.csv"),
    ("A・暗い", "20261117_021829_946788", "fold_cup_021829_946788", "cup_20261117_021829_946788", "cup_20261117_021829_946788.csv"),
    ("B・明るい", "20261117_022355_294680", "fold_cup_022355_294680", "cup_20261117_022355_294680", "cup_20261117_022355_294680.csv"),
    ("B・暗い（評価用）", "20261117_022205_950242", "test_cup_v5", "cup_20261117_022205_950242", "cup_20261117_022205_950242.csv"),
]
VALIDATION = ("11/10 手持ち（調整用）", "20261110_094730_024772", "val_cup_v1", "cup_20261110_run02",
              "cup_20261110_094730.csv")


def evaluation(run: str) -> dict[str, dict]:
    data = json.loads((RESEARCH / run / "evaluation.json").read_text(encoding="utf-8"))
    return {g["method"]: g for g in data["groups"]}


def speed(run: str) -> dict[str, float]:
    data = json.loads((RESEARCH / run / "run.json").read_text(encoding="utf-8"))
    return {m: v["compute_fps"] for m, v in data["summary"].items()}


def breakdown(label, session, run, candidates_run, annotation_file, learned_run=None) -> dict:
    """Visible reviewed frames split by outcome. Candidates were measured on a
    detector-only history, so 'reachable' is what an ideal choice could add there."""
    truth = {f: a for (s, f), a in load_annotations(ANNOTATIONS / annotation_file).items()
             if a["reviewed"] and a["visible"]}
    groups = evaluation(run)
    learned = (evaluation(learned_run) if learned_run else groups)["learned"]["detection"]["true_positives"]
    base = groups["baseline"]["detection"]["true_positives"]
    reachable = tried = unavailable = 0
    with (RESEARCH / candidates_run / "candidates.jsonl").open(encoding="utf-8") as stream:
        for line in stream:
            row = json.loads(line)
            a = truth.get(row["frame_id"])
            if a is None:
                continue
            hits = [c for c in row["candidates"] if c["detection"] and box_iou(c["detection"], a) >= 0.5]
            if any(c["kind"] == "none" for c in hits):
                continue  # the detector alone already found it
            if hits:
                reachable += 1
            elif row["target_available"]:
                tried += 1
            else:
                unavailable += 1
    added = learned - base
    return {"label": label, "visible": len(truth), "base": base, "added": added,
            "missed_choice": max(0, reachable - added), "tried": tried, "unavailable": unavailable}


def fig_tradeoff(plt, out: Path) -> None:
    totals = {m: [0, 0] for m in METHOD_NAMES}
    fps = {m: [] for m in METHOD_NAMES}
    for _, _, run, _, _ in SESSIONS:
        groups, speeds = evaluation(run), speed(run)
        for m in METHOD_NAMES:
            d = groups[m]["detection"]
            totals[m][0] += d["true_positives"]
            totals[m][1] += d["true_positives"] + d["false_negatives"]
            fps[m].append(speeds[m])
    fig, ax = plt.subplots(figsize=(9, 5.4))
    offsets = {"baseline": (-10, 8, "right"), "bicubic": (10, -4, "left"), "fixed": (10, -14, "left"),
               "rule": (10, 4, "left"), "learned": (10, 8, "left")}
    for m, name in METHOD_NAMES.items():
        x, y = float(np.mean(fps[m])), totals[m][0] / totals[m][1]
        highlight = m == "learned"
        ax.scatter([x], [y], s=110 if highlight else 80, color=BLUE if highlight else GRAY, edgecolor=SURFACE,
                   linewidth=2, zorder=3)
        dx, dy, ha = offsets[m]
        ax.annotate(f"{name}\n{y:.1%}・{x:.0f} fps", (x, y), xytext=(dx, dy), textcoords="offset points",
                    ha=ha, va="center", fontsize=10.5 if highlight else 10, color=INK if highlight else INK2,
                    fontweight="bold" if highlight else "normal")
    ax.set_xlabel("処理速度 (fps, このPCのGPU、4本の平均)")
    ax.set_ylabel("検出率（見えているフレームのうち）")
    ax.set_xlim(20, 80)
    ax.set_ylim(0.36, 0.455)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.grid(color=GRID, lw=0.8)
    ax.set_title("右上ほど良い：学習による選択は、超解像と同じ精度を通常拡大並みの速さで出す", loc="left",
                 fontsize=13, color=INK, pad=10)
    n = totals["baseline"][1]
    fig.text(0.01, -0.03, f"11/17撮影の4本（カメラ2位置×明暗2段階）で、1本ずつ評価に回す交差検証の合計（見えているフレーム{n}枚）。"
             "各本は、その本を学習に使っていない選択モデルで評価。\n誤検出はすべての方式で0件（カップのない318枚）。"
             "超解像を使う3方式の差は±3枚で、精度の差はないとみるのが妥当。", fontsize=8.5, color=MUTED)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_breakdown(plt, rows: list[dict], out: Path) -> None:
    parts = [("base", "通常YOLOで検出", "#184f95", None),
             ("added", "学習による選択で追加", "#3987e5", None),
             ("missed_choice", "候補のどれかなら検出できた", "#9ec5f4", None),
             ("tried", "切り出して拡大しても検出できず", "#d9d8d4", None),
             ("unavailable", "見失っていて切り出す場所がない", "#f0efec", "//")]
    fig, ax = plt.subplots(figsize=(12, 4.8))
    for i, row in enumerate(rows):
        y = len(rows) - 1 - i
        left = 0.0
        for key, name, color, hatch in parts:
            share = row[key] / row["visible"]
            ax.barh(y, share, left=left, height=0.62, color=color, edgecolor=SURFACE if not hatch else "#b5b4ae",
                    linewidth=2 if not hatch else 0, hatch=hatch, label=name if i == 0 else None)
            if share >= 0.045:
                dark = key in ("base", "added")
                ax.text(left + share / 2, y, f"{share:.0%}", ha="center", va="center", fontsize=10,
                        color="white" if dark else INK, fontweight="bold" if key in ("added", "unavailable") else "normal")
            left += share
        ax.text(-0.01, y, f"{row['label']}\n（{row['visible']}枚）", ha="right", va="center", fontsize=10.5, color=INK)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.6, len(rows) - 0.4)
    ax.set_yticks([])
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.spines["left"].set_visible(False)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=5, fontsize=9.5,
              handlelength=1.4, columnspacing=1.2)
    ax.set_title("見えているカップの行き先：拡大の上乗せは条件により2〜19%。残りの多くは「見失った後に探す場所がない」",
                 loc="left", fontsize=13, color=INK, pad=10)
    fig.text(0.01, -0.13, "A/B＝カメラ位置。「候補のどれか」と「切り出す場所がない」は、通常YOLOだけで追跡した場合の全24候補の結果から集計"
             "（見失って0.75秒を過ぎると切り出す位置の手がかりがなくなる）。\n11/10は学習した選択の設定を決めるのに使った映像。"
             "11/17の4本は、それぞれを学習に使っていない選択モデルの結果。", fontsize=8.5, color=MUTED)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--recording", type=Path, required=True, help="held-out dark recording for fig7")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    plt = progress.setup_matplotlib()

    fig_tradeoff(plt, args.output / "fig5_tradeoff.png")
    rows = [breakdown(*VALIDATION, learned_run="val_cup_v5")] + [breakdown(*s) for s in SESSIONS]
    for row in rows:
        print(row)
    fig_breakdown(plt, rows, args.output / "fig6_breakdown.png")

    # held-out dark session: median-confidence frame among those only the enhancement found
    label, session, run, _, annotation_file = SESSIONS[3]
    preds = progress.load_predictions(RESEARCH / run)
    truth = {f: a for (s, f), a in load_annotations(ANNOTATIONS / annotation_file).items()
             if a["reviewed"] and a["visible"]}

    def found(method, f):
        box = progress.box_of(preds[method][f])
        return box is not None and box_iou(dict(zip(("x", "y", "width", "height"), box)), truth[f]) >= 0.5
    eligible = sorted((float(preds["fixed"][f]["confidence"]), f) for f in truth
                      if not found("baseline", f) and not found("bicubic", f) and found("fixed", f))
    pick = eligible[len(eligible) // 2][1]
    print(f"fig7 frame {pick}: median confidence of {len(eligible)} frames")
    box = tuple(int(float(truth[pick][k])) for k in ("x", "y", "width", "height"))
    rows_csv = list(csv.DictReader((args.recording / "frames.csv").open(encoding="utf-8-sig")))
    fps = (len(rows_csv) - 1) / float(rows_csv[-1]["timestamp_s"])
    progress.fig_inputs(plt, args.recording, preds, box, pick, ROOT / "yolo26n.pt", args.output / "fig7_dark_example.png",
                        fps=fps)
    print(f"written to {args.output.resolve()}")


if __name__ == "__main__":
    main()
