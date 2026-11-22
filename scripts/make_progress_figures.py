"""Progress figures and comparison videos for the advisor meeting (2026-11).

Inputs are a replay run with --collect-candidates (predictions.csv,
candidates.jsonl), the matching recording, draft proposals (only for the
object-size curve) and optionally a second run for another object. Outputs:
- video_static.mp4 / video_full.mp4: the four methods side by side
- fig1_timeline.png: object size and per-method detection over time
- fig2_heatmap.png: re-detection rate per crop padding x enlargement
- fig3_inputs.png: what the detector sees for crop-only / bicubic / FSRCNN
- fig4_pareto.png: re-detection rate vs extra time per candidate

Rates are "frames where the method reported the object". They are checked
against the object's fixed position only inside --static (object at rest).
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from small_object_lab.metrics import Detection  # noqa: E402
from small_object_lab.super_resolution import FSRCNNSuperResolution, expanded_crop_region  # noqa: E402

FONT = "C:/Windows/Fonts/BIZ-UDGothicR.ttc"
FONT_BOLD = "C:/Windows/Fonts/BIZ-UDGothicB.ttc"
METHODS = [("baseline", "通常YOLO"), ("bicubic", "通常拡大（bicubic x4）"),
           ("fixed", "局所超解像（FSRCNN x4）"), ("rule", "動的倍率規則")]
INK, INK2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1", "#fcfcfb"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
PADDINGS = [0.1, 0.2, 0.5, 1.0, 2.0, 3.0]
KINDS = [("crop", 1, "切り出しのみ"), ("sr", 2, "超解像 x2"), ("sr", 4, "超解像 x4"), ("sr", 8, "超解像 x8")]


def iou(a, b) -> float:
    ix = max(0, min(a[0] + a[2], b[0] + b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[1] + a[3], b[1] + b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / (a[2] * a[3] + b[2] * b[3] - inter)


def load_predictions(run: Path) -> dict[str, dict[int, dict]]:
    by = collections.defaultdict(dict)
    with (run / "predictions.csv").open(encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            by[row["method"]][int(row["frame_id"])] = row
    return by


def box_of(row) -> tuple[int, int, int, int] | None:
    return tuple(int(row[k]) for k in ("x", "y", "width", "height")) if row["observed"] == "1" else None


# ---------------------------------------------------------------- video
def put_text(image: np.ndarray, items: list[tuple[str, tuple[int, int], int, tuple[int, int, int], bool]]) -> np.ndarray:
    from PIL import Image, ImageDraw, ImageFont
    pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    for text, xy, size, rgb, bold in items:
        draw.text(xy, text, font=ImageFont.truetype(FONT_BOLD if bold else FONT, size), fill=rgb)
    return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)


def make_video(recording: Path, preds, frames: list[int], out: Path, view: tuple[int, int, int, int],
               panel: int, fps: float, title: str) -> None:
    vx, vy, vw, vh = view
    gain = panel / vw
    panel_h = round(vh * gain)
    header, strip = 60, 22
    width, height = panel * 4 + 18, header + panel_h + strip + 40
    writer = None
    for backend, code in ((cv2.CAP_MSMF, "H264"), (cv2.CAP_ANY, "avc1"), (cv2.CAP_ANY, "mp4v")):
        writer = cv2.VideoWriter(str(out), backend, cv2.VideoWriter_fourcc(*code), fps, (width, height))
        if writer.isOpened():
            print(f"{out.name}: {code}")
            break
    found = collections.Counter()
    for index, frame_id in enumerate(frames):
        image = cv2.imread(str(recording / f"frames/{frame_id:08d}.png"))
        canvas = np.full((height, width, 3), 252, np.uint8)
        texts = [(f"{title}   {frame_id / fps:5.1f} 秒", (10, 8), 22, (11, 11, 11), True)]
        for slot, (method, name) in enumerate(METHODS):
            ox = slot * (panel + 6)
            oy = 40
            view_image = cv2.resize(image[vy:vy + vh, vx:vx + vw], (panel, panel_h), interpolation=cv2.INTER_AREA)
            box = box_of(preds[method][frame_id])
            found[method] += box is not None
            if box:
                x, y, w, h = box
                p1 = (round((x - vx) * gain), round((y - vy) * gain))
                p2 = (round((x + w - vx) * gain), round((y + h - vy) * gain))
                cv2.rectangle(view_image, p1, p2, (40, 200, 40), 3)
            canvas[oy + header:oy + header + panel_h, ox:ox + panel] = view_image
            # detection history strip: dark = found, light = missed
            sy = oy + header + panel_h + 4
            for k, f in enumerate(frames[:index + 1]):
                x0 = ox + round(k * panel / len(frames))
                x1 = ox + max(round((k + 1) * panel / len(frames)), x0 + 1)
                hit = preds[method][f]["observed"] == "1"
                canvas[sy:sy + strip - 6, x0:x1] = (191, 106, 37) if hit else (225, 229, 230)
            rate = found[method] / (index + 1)
            status = ("検出", (30, 140, 30)) if box else ("見失い", (200, 40, 40))
            texts += [(name, (ox + 6, oy + 6), 19, (11, 11, 11), True),
                      (f"{status[0]}　累計 {rate:.0%}", (ox + 6, oy + 32), 18, status[1], True)]
        writer.write(put_text(canvas, texts))
    writer.release()


# ---------------------------------------------------------------- figures
def setup_matplotlib():
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import font_manager
    import matplotlib.pyplot as plt
    font_manager.fontManager.addfont(FONT)
    name = font_manager.FontProperties(fname=FONT).get_name()
    plt.rcParams.update({"font.family": name, "font.size": 11, "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
                         "savefig.facecolor": SURFACE})
    return plt


def fig_timeline(plt, preds, proposals: Path, fps: float, sections, static, out: Path) -> None:
    sizes = {}
    with proposals.open(encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            sizes[int(row["frame_id"])] = max(int(row["width"]), int(row["height"]))
    n = max(preds["baseline"])
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(12, 5.4), sharex=True,
                                      gridspec_kw={"height_ratios": [1, 1.5], "hspace": 0.12})
    t = np.array(sorted(sizes)) / fps
    top.plot(t, [sizes[f] for f in sorted(sizes)], ".", ms=2.5, color=INK2)
    top.set_yscale("log")
    top.set_yticks([20, 50, 100, 200])
    top.set_yticklabels(["20", "50", "100", "200"])
    top.yaxis.set_minor_formatter(plt.NullFormatter())
    top.set_ylim(18, 420)
    top.set_ylabel("カップの長辺 (px)")
    top.grid(axis="y", color=GRID, lw=0.8)
    for (a, b, label) in sections:
        for ax in (top, bottom):
            ax.axvspan(a / fps, b / fps, color="#f0efec", zorder=0)
        top.text((a + b) / 2 / fps, 380, label, ha="center", va="top", color=INK2, fontsize=10)
    for row, (method, name) in enumerate(METHODS):
        y = len(METHODS) - 1 - row
        hits = [f for f in range(1, n + 1) if preds[method][f]["observed"] == "1"]
        bottom.broken_barh([(f / fps, 1 / fps) for f in hits], (y - 0.32, 0.64), color=BLUE_RAMP[4])
        a, b = static
        s = sum(preds[method][f]["observed"] == "1" for f in range(a, b + 1)) / (b - a + 1)
        bottom.text(n / fps + 0.6, y, f"{name}\n全体 {len(hits) / n:.0%} / 静止区間 {s:.0%}", va="center",
                    fontsize=9.5, color=INK)
    bottom.set_yticks([])
    bottom.spines["left"].set_visible(False)
    bottom.set_xlabel("時間 (秒)")
    bottom.set_xlim(0, n / fps)
    top.set_title("カップが小さくなるほど方式の差が開く（濃い帯＝そのフレームで検出）", loc="left", fontsize=13,
                  color=INK, pad=10)
    fig.text(0.01, 0.005, "静止区間はカップの固定位置と照合済み（誤検出0件）。それ以外の区間は正解未確認の検出割合。"
             "カップの大きさは大きい検出器（yolo26x/l）による下書きから。", fontsize=8.5, color=MUTED)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def candidate_rates(path: Path, keep) -> tuple[int, dict, dict]:
    samples = [json.loads(line) for line in path.open(encoding="utf-8")]
    chosen = [s for s in samples if keep(s) and s["target_available"] and s["candidates"][0]["detection"] is None]
    hit, ms = collections.Counter(), collections.defaultdict(list)
    for s in chosen:
        for c in s["candidates"][1:]:
            hit[c["action_id"]] += bool(c["adopted"])
            ms[c["action_id"]].append(c["elapsed_ms"])
    rates = {a: hit[a] / len(chosen) for a in ms}
    return len(chosen), rates, {a: float(np.mean(v)) for a, v in ms.items()}


def action_id(kind: str, padding: float, scale: int) -> str:
    return f"{kind}_p{round(padding * 100):03d}_x{scale}"


def fig_heatmap(plt, panels, out: Path) -> None:
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("blue", BLUE_RAMP)
    fig, axes = plt.subplots(1, len(panels), figsize=(6.2 * len(panels), 5.2))
    axes = np.atleast_1d(axes)
    for ax, (title, n, rates, ms) in zip(axes, panels):
        grid = np.full((len(PADDINGS), len(KINDS)), np.nan)
        for i, p in enumerate(PADDINGS):
            for j, (kind, scale, _) in enumerate(KINDS):
                a = action_id(kind, p, scale)
                if a in rates:
                    grid[i, j] = rates[a]
        ax.imshow(np.ma.masked_invalid(grid), cmap=cmap, vmin=0, vmax=1, aspect="auto")
        best = np.unravel_index(np.nanargmax(grid), grid.shape)
        for i, p in enumerate(PADDINGS):
            for j, (kind, scale, _) in enumerate(KINDS):
                a = action_id(kind, p, scale)
                if a not in rates:
                    ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, hatch="//", ec=GRID, lw=0))
                    ax.text(j, i, "対象外", ha="center", va="center", fontsize=8.5, color=MUTED)
                    continue
                dark = rates[a] >= 0.5
                ax.text(j, i - 0.12, f"{rates[a]:.0%}", ha="center", va="center", fontsize=13,
                        color="white" if dark else INK, fontweight="bold")
                ax.text(j, i + 0.25, f"{ms[a]:.0f} ms", ha="center", va="center", fontsize=8.5,
                        color="white" if dark else INK2)
        ax.add_patch(plt.Rectangle((best[1] - 0.47, best[0] - 0.47), 0.94, 0.94, fill=False, ec="#eb6834", lw=3))
        ax.set_xticks(range(len(KINDS)), [k[2] for k in KINDS])
        ax.set_yticks(range(len(PADDINGS)), [f"{round(p * 100)}%" for p in PADDINGS])
        ax.set_ylabel("切り出しの余白（各辺）")
        ax.set_title(f"{title}\n通常YOLOが見失った {n} フレーム", loc="left", fontsize=12, color=INK)
        ax.tick_params(length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.suptitle("見つけ直せた割合：最適な「余白×拡大方法」は物によって違う（橙枠＝最良）", x=0.01, ha="left",
                 fontsize=13.5, color=INK)
    fig.text(0.01, -0.02, "処理時間は候補1回あたりの平均（このPCのGPU）。対象外＝拡大後640pxを超えるため候補から除外。"
             "カップは固定位置と照合済み、鍵は10/27の映像で正解未確認。", fontsize=8.5, color=MUTED)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_pareto(plt, n, rates, ms, out: Path, fixed_action: str) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    groups = [("切り出しのみ", "crop", (1,), "o"), ("超解像 x2", "sr", (2,), "s"), ("超解像 x4・x8", "sr", (4, 8), "^")]
    best = max(rates, key=lambda a: (rates[a], -ms[a]))
    callouts = {best, fixed_action}
    for color, (name, kind, scales, marker) in zip(SERIES, groups):
        points = [(ms[a], rates[a], a) for a in rates if a.startswith(kind) and int(a.split("_x")[1]) in scales]
        ax.scatter([p[0] for p in points], [p[1] for p in points], s=70, color=color, marker=marker,
                   edgecolor=SURFACE, linewidth=2, label=name, zorder=3)
        merged: list[list] = []  # points that sit on top of each other share one label
        for x, y, a in sorted(points, key=lambda p: p[2]):
            padding, scale = int(a.split("_p")[1][:3]), a.split("_x")[1]
            suffix = f"・x{scale}" if kind == "sr" and len(scales) > 1 else ""
            if a in callouts:
                continue
            for item in merged:
                if abs(item[0] - x) < 1.5 and abs(item[1] - y) < 0.03 and item[3] == suffix:
                    item[2].append(padding)
                    break
            else:
                merged.append([x, y, [padding], suffix])
        for x, y, paddings, suffix in merged:
            text = "/".join(str(p) for p in sorted(paddings)) + "%" + suffix
            ax.annotate(text, (x, y), xytext=(7, 4), textcoords="offset points", fontsize=8.5, color=INK2)
    ax.annotate(f"余白{int(best.split('_p')[1][:3])}%・超解像x{best.split('_x')[1]}\n{rates[best]:.0%}・{ms[best]:.0f} ms",
                (ms[best], rates[best]), xytext=(-14, -42), textcoords="offset points", fontsize=10.5, color=INK,
                ha="right", arrowprops=dict(arrowstyle="-", color=INK2, lw=1))
    ax.annotate(f"固定方式の設定（余白20%・x4）\n{rates[fixed_action]:.0%}・{ms[fixed_action]:.0f} ms",
                (ms[fixed_action], rates[fixed_action]), xytext=(24, -36), textcoords="offset points", fontsize=10,
                color=INK2, arrowprops=dict(arrowstyle="-", color=INK2, lw=1))
    ax.set_xlabel("候補1回あたりの追加処理時間 (ms, このPCのGPU)　ラベルは余白（各辺）")
    ax.set_ylabel("見つけ直せた割合")
    ax.set_ylim(-0.04, 1.08)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}"))
    ax.grid(color=GRID, lw=0.8)
    ax.legend(frameon=False, loc="upper right")
    ax.set_title(f"左上ほど「速くて見つかる」：机の上のカップ（通常YOLOが見失った{n}フレーム）", loc="left",
                 fontsize=13, color=INK)
    fig.text(0.01, -0.02, "低スペック機器での処理時間は今後計測予定。選択モデルは、状況ごとに左上の候補を選ぶことを狙う。",
             fontsize=8.5, color=MUTED)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_inputs(plt, recording: Path, preds, truth, frame_id: int, model_path: Path, out: Path,
               padding: float = 0.2, scale: int = 4, fps: float = 16.8) -> None:
    from ultralytics import YOLO
    model = YOLO(str(model_path))
    sr = FSRCNNSuperResolution()
    image = cv2.imread(str(recording / f"frames/{frame_id:08d}.png"))
    target = Detection("cup", 1, *truth)
    region = expanded_crop_region(target, image.shape[1], image.shape[0], padding_ratio=padding)
    crop = image[region.y:region.y + region.height, region.x:region.x + region.width]
    local = (truth[0] - region.x, truth[1] - region.y, truth[2], truth[3])
    variants = [("切り出しのみ", crop, 1.0),
                (f"通常拡大（bicubic x{scale}）", cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC), scale),
                (f"超解像（FSRCNN x{scale}）", sr.enhance(crop, scale=scale), scale)]
    fig, axes = plt.subplots(1, 4, figsize=(14, 4.4), gridspec_kw={"width_ratios": [1.5, 1, 1, 1]})
    thumb = image.copy()
    cv2.rectangle(thumb, (region.x, region.y), (region.x + region.width, region.y + region.height), (52, 104, 235), 3)
    axes[0].imshow(cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB))
    axes[0].set_title(f"元画像（{frame_id / fps:.1f}秒）と切り出し範囲\n通常YOLO：見失い", loc="left", fontsize=11, color=INK)
    for ax, (name, enlarged, gain) in zip(axes[1:], variants):
        # what YOLO sees: its own linear resize to 640 on the long side
        side = 640 / max(enlarged.shape[:2])
        seen = cv2.resize(enlarged, (round(enlarged.shape[1] * side), round(enlarged.shape[0] * side)),
                          interpolation=cv2.INTER_LINEAR)
        result = model.predict(enlarged, imgsz=640, conf=0.05, rect=False, device=0, verbose=False)[0]
        best = 0.0
        for (x1, y1, x2, y2), conf, cls in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy(),
                                               result.boxes.cls.cpu().numpy()):
            if model.names[int(cls)] != "cup":
                continue
            box = (x1 / gain, y1 / gain, (x2 - x1) / gain, (y2 - y1) / gain)
            if iou(box, local) >= 0.5 and conf > best:
                best = float(conf)
                drawn = tuple(round(v * gain * side) for v in box)
        if best >= 0.2:
            cv2.rectangle(seen, drawn[:2], (drawn[0] + drawn[2], drawn[1] + drawn[3]), (40, 200, 40), 3)
        ax.imshow(cv2.cvtColor(seen, cv2.COLOR_BGR2RGB))
        verdict = f"検出（信頼度 {best:.2f}）" if best >= 0.2 else f"見失い（信頼度 {best:.2f}）"
        ax.set_title(f"{name}\n→ {verdict}", loc="left", fontsize=11, color="#1a7f1a" if best >= 0.2 else "#c03030")
    for ax in axes:
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.suptitle("検出器が実際に見る画像（同じ範囲を640pxに拡大）：超解像で輪郭がはっきりする", x=0.01, ha="left",
                 fontsize=13.5, color=INK)
    fig.text(0.01, -0.03, f"超解像だけが見つけたフレームのうち、信頼度が中央値のフレーム。余白{round(padding * 100)}%・x{scale}は固定方式の設定。拡大後の画像を検出器（yolo26n）が640pxに線形補間で拡大し直す。"
             "閾値0.20。", fontsize=8.5, color=MUTED)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run", type=Path)
    parser.add_argument("recording", type=Path)
    parser.add_argument("proposals", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--static", type=int, nargs=2, required=True, help="first/last frame with the object at rest")
    parser.add_argument("--static-box", type=int, nargs=4, required=True, help="x y w h of the object at rest")
    parser.add_argument("--sections", nargs="+", default=[], help="first:last:label")
    parser.add_argument("--compare-run", type=Path, help="Second object's candidates run (heatmap)")
    parser.add_argument("--compare-title", default="鍵")
    parser.add_argument("--compare-min-time", type=float, default=30.0)
    parser.add_argument("--video-start", type=int, default=1)
    parser.add_argument("--model", type=Path, default=Path(__file__).resolve().parents[1] / "yolo26n.pt")
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)
    preds = load_predictions(args.run)
    rows = list(csv.DictReader((args.recording / "frames.csv").open(encoding="utf-8-sig")))
    fps = (len(rows) - 1) / float(rows[-1]["timestamp_s"])
    a, b = args.static
    gt = tuple(args.static_box)
    for method, name in METHODS:
        ok = sum(1 for f in range(a, b + 1) if (box := box_of(preds[method][f])) and iou(box, gt) >= 0.5)
        print(f"static {name}: {ok}/{b - a + 1}")

    plt = setup_matplotlib()
    sections = [(int(s.split(":")[0]), int(s.split(":")[1]), s.split(":", 2)[2]) for s in args.sections]
    fig_timeline(plt, preds, args.proposals, fps, sections, (a, b), args.output / "fig1_timeline.png")

    n_cup, cup_rates, cup_ms = candidate_rates(args.run / "candidates.jsonl", lambda s: a <= s["frame_id"] <= b)
    panels = [("黒いカップ（机の上に静止、幅33px）", n_cup, cup_rates, cup_ms)]
    if args.compare_run:
        n_key, key_rates, key_ms = candidate_rates(args.compare_run / "candidates.jsonl",
                                                   lambda s: s["timestamp_s"] >= args.compare_min_time)
        panels.append((f"{args.compare_title}（部屋の奥、幅24〜32px）", n_key, key_rates, key_ms))
    fig_heatmap(plt, panels, args.output / "fig2_heatmap.png")
    fig_pareto(plt, n_cup, cup_rates, cup_ms, args.output / "fig4_pareto.png", "sr_p020_x4")

    # a static frame where baseline and bicubic missed but FSRCNN found the cup
    eligible = sorted((float(preds["fixed"][f]["confidence"]), f) for f in range(a, b + 1)
                      if preds["baseline"][f]["observed"] == "0" and preds["bicubic"][f]["observed"] == "0"
                      and preds["fixed"][f]["observed"] == "1")
    pick = eligible[len(eligible) // 2][1]
    print(f"fig3 frame {pick}: median FSRCNN confidence {eligible[len(eligible) // 2][0]:.2f} of {len(eligible)} frames")
    fig_inputs(plt, args.recording, preds, gt, pick, args.model, args.output / "fig3_inputs.png")

    pad = 150
    height, width = cv2.imread(str(args.recording / rows[0]["path"])).shape[:2]
    side = min(2 * pad + max(gt[2], gt[3]), width, height)
    view = (int(min(max(0, gt[0] + gt[2] / 2 - side / 2), width - side)),
            int(min(max(0, gt[1] + gt[3] / 2 - side / 2), height - side)), side, side)
    first, last = max(1, a - 30), min(max(preds["baseline"]), b + 15)
    make_video(args.recording, preds, list(range(first, last + 1)), args.output / "video_static.mp4",
               view, 400, fps, "机の上のカップ（拡大表示）")
    make_video(args.recording, preds, list(range(args.video_start, max(preds["baseline"]) + 1)),
               args.output / "video_full.mp4", (0, 0, 720, 720), 320, fps, "全体")
    print(f"written to {args.output.resolve()}")


if __name__ == "__main__":
    main()
