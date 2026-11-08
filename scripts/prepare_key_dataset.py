"""Build the single-class key dataset from Objects365 (val release) and LeftInCAR.

Only key-related files are kept locally:
- Objects365 v2 val: the official annotation JSON is downloaded, then each image
  patch archive is streamed and only images with a "Key" box (plus a few
  key-free backgrounds) are extracted. Archives are never stored. Objects365 is
  for academic purposes only; images must not be redistributed.
- LeftInCAR (CC BY 4.0, DOI 10.17632/dd4hw3r6dh.1): the 8.4 GB zip is read with
  HTTP range requests, so only labels and the selected images are transferred.

Subcommands: fetch-objects365, fetch-leftincar, build. Fetches are resumable.

Example:
    .venv\\Scripts\\python.exe scripts\\prepare_key_dataset.py fetch-objects365
    .venv\\Scripts\\python.exe scripts\\prepare_key_dataset.py fetch-leftincar
    .venv\\Scripts\\python.exe scripts\\prepare_key_dataset.py build --output datasets/key_o365_leftincar
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
import re
import shutil
import sys
import tarfile
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # for small_object_lab (FSRCNN)

# Same bucket as Ultralytics' Objects365.yaml; the ksyuncs.com host measured faster.
# Each connection is throttled (~0.1-0.3 MB/s), so transfers run in parallel.
O365_BASE = "https://dorc.ks3-cn-beijing.ksyuncs.com/data-set/2020Objects365%E6%95%B0%E6%8D%AE%E9%9B%86/val/"
O365_DIR = Path("datasets/objects365")
LEFTINCAR_URL = ("https://data.mendeley.com/public-files/datasets/dd4hw3r6dh/files/"
                 "380abbd8-2119-41fe-9157-c69cf0a0a4f7/file_downloaded")
LEFTINCAR_SHA256 = "3b7904ba71650c42a1c8d9c7a6257a35e75c8c0716094afdcac9408155ddf9a0"
LEFTINCAR_KEYS_CLASS = 7  # smartphone, laptop, card, suitcase, wallet, backpack, clothing, keys, glasses, handbag
LEFTINCAR_DIR = Path("datasets/leftincar")
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp")  # LeftInCAR ships uncompressed BMP


def log(message: str) -> None:
    print(f"[{datetime.now():%H:%M:%S}] {message}", flush=True)


def download(url: str, target: Path, parts: int = 8) -> Path:
    """Download with parallel HTTP ranges; the result appears only when complete."""
    if target.exists():
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(urllib.request.Request(url, headers={"Range": "bytes=0-0"}), timeout=60) as response:
        size = int(response.headers["Content-Range"].rsplit("/", 1)[1])
    step = -(-size // parts)
    pieces = [(i, i * step, min(size, (i + 1) * step) - 1) for i in range(parts) if i * step < size]

    def fetch(piece: tuple[int, int, int]) -> Path:
        index, start, end = piece
        path = target.with_name(f"{target.name}.part{index}")
        for attempt in range(3):
            try:
                request = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
                with urllib.request.urlopen(request, timeout=120) as response, path.open("wb") as stream:
                    shutil.copyfileobj(response, stream, 1 << 20)
                if path.stat().st_size == end - start + 1:
                    return path
            except OSError:
                if attempt == 2:
                    raise
            time.sleep(3)
        raise OSError(f"Incomplete range {start}-{end} of {url}")

    with ThreadPoolExecutor(len(pieces)) as pool:
        paths = list(pool.map(fetch, pieces))
    temporary = target.with_name(target.name + ".download")
    with temporary.open("wb") as stream:
        for path in paths:
            with path.open("rb") as part:
                shutil.copyfileobj(part, stream, 1 << 20)
            path.unlink()
    temporary.replace(target)
    return target


# --------------------------------------------------------------------------- Objects365
def o365_selection(annotation: dict, negative_ratio: float, seed: int) -> dict:
    key_ids = {c["id"] for c in annotation["categories"] if c["name"].strip().casefold() == "key"}
    if len(key_ids) != 1:
        raise ValueError(f"Expected one Objects365 'Key' category, found {key_ids}")
    boxes: dict[int, list[dict]] = {}
    for item in annotation["annotations"]:
        if item["category_id"] in key_ids:
            boxes.setdefault(item["image_id"], []).append(item)
    # Drawings/prints of keys (isfake) are not physical keys; drop those whole images.
    fake = {image_id for image_id, items in boxes.items() if any(i.get("isfake") for i in items)}
    positives = sorted(set(boxes) - fake)
    others = sorted({image["id"] for image in annotation["images"]} - set(boxes))
    negatives = sorted(random.Random(seed).sample(others, round(len(positives) * negative_ratio)))
    return {"key_category_ids": sorted(key_ids), "positives": positives, "negatives": negatives,
            "excluded_fake": sorted(fake), "boxes": boxes}


def fetch_objects365(args: argparse.Namespace) -> None:
    annotation_path = download(O365_BASE + "zhiyuan_objv2_val.json", O365_DIR / "annotations" / "zhiyuan_objv2_val.json")
    log("Loading Objects365 val annotations")
    annotation = json.loads(annotation_path.read_text(encoding="utf-8"))
    selection = o365_selection(annotation, args.negative_ratio, args.seed)
    images = {image["id"]: image for image in annotation["images"]}
    wanted = {Path(images[i]["file_name"]).name: i for i in [*selection["positives"], *selection["negatives"]]}
    patches: dict[str, set[str]] = {}
    for name, image_id in wanted.items():
        file_name = images[image_id]["file_name"]  # e.g. images/v1/patch0/objects365_v1_00000926.jpg
        patches.setdefault(str(Path(file_name).parent).replace("\\", "/"), set()).add(name)
    image_dir = O365_DIR / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    log(f"{len(selection['positives'])} key images, {len(selection['negatives'])} backgrounds in {len(patches)} patches")

    def extract(patch: str, names: set[str]) -> list[str]:
        missing = {n for n in names if not (image_dir / n).exists()}
        if not missing:
            return []
        url = O365_BASE + patch + ".tar.gz"
        for attempt in range(3):
            try:
                # Stream the archive; only the wanted JPEGs ever touch the disk.
                with urllib.request.urlopen(url, timeout=120) as response, \
                        tarfile.open(fileobj=response, mode="r|gz") as archive:
                    for member in archive:
                        name = Path(member.name).name
                        if member.isfile() and name in missing:
                            temporary = image_dir / (name + ".part")
                            temporary.write_bytes(archive.extractfile(member).read())
                            temporary.replace(image_dir / name)
                break
            except (OSError, tarfile.TarError) as error:
                if attempt == 2:
                    raise
                log(f"{patch}: retry after {error}")
                time.sleep(5)
        still = sorted(n for n in missing if not (image_dir / n).exists())
        log(f"{patch}: extracted {len(missing) - len(still)}, missing {len(still)}")
        return still

    with ThreadPoolExecutor(args.workers) as pool:
        missing = [name for result in pool.map(lambda item: extract(*item), patches.items()) for name in result]
    if missing:
        raise RuntimeError(f"{len(missing)} files not found in their archives, e.g. {missing[:3]}")
    log("Objects365 fetch complete")


# --------------------------------------------------------------------------- LeftInCAR
class HttpRangeFile(io.RawIOBase):
    """Seekable read-only view of a remote file using HTTP Range requests."""

    HEADERS = {"User-Agent": "YOLO26SmallObjectLab/0.1 (research dataset preparation)"}

    def __init__(self, url: str) -> None:
        # Mendeley rejects HEAD; a one-byte range GET also follows its redirect.
        request = urllib.request.Request(url, headers={**self.HEADERS, "Range": "bytes=0-0"})
        with urllib.request.urlopen(request, timeout=60) as response:
            self.url = response.geturl()
            content_range = response.headers.get("Content-Range", "")
            if response.status != 206 or "/" not in content_range:
                raise OSError("Server does not support range requests")
            self.size = int(content_range.rsplit("/", 1)[1])
        self.position = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        base = {io.SEEK_SET: 0, io.SEEK_CUR: self.position, io.SEEK_END: self.size}[whence]
        self.position = max(0, base + offset)
        return self.position

    def readinto(self, buffer) -> int:
        if self.position >= self.size or not len(buffer):
            return 0
        end = min(self.size, self.position + len(buffer)) - 1
        request = urllib.request.Request(self.url, headers={**self.HEADERS, "Range": f"bytes={self.position}-{end}"})
        for attempt in range(3):
            try:
                with urllib.request.urlopen(request, timeout=120) as response:
                    data = response.read()
                break
            except OSError:
                if attempt == 2:
                    raise
                time.sleep(3)
        buffer[: len(data)] = data
        self.position += len(data)
        return len(data)


def fetch_leftincar(args: argparse.Namespace) -> None:
    remote = io.BufferedReader(HttpRangeFile(LEFTINCAR_URL), buffer_size=1 << 20)
    with zipfile.ZipFile(remote) as archive:
        # The zip also carries macOS resource forks (__MACOSX/, ._*); ignore them.
        names = [n for n in archive.namelist()
                 if not n.startswith("__MACOSX/") and not Path(n).name.startswith("._")]
        labels = {Path(n).stem: n for n in names if n.startswith("labels/") and n.endswith(".txt")}
        images = {Path(n).stem: n for n in names if n.startswith("images/") and n.lower().endswith(IMAGE_SUFFIXES)}
        log(f"LeftInCAR zip: {len(images)} images, {len(labels)} label files")
        label_dir, image_dir = LEFTINCAR_DIR / "labels", LEFTINCAR_DIR / "images"
        label_dir.mkdir(parents=True, exist_ok=True)
        image_dir.mkdir(parents=True, exist_ok=True)
        with_keys, without_keys = [], []
        for stem in sorted(images):
            text = archive.read(labels[stem]).decode("utf-8") if stem in labels else ""
            (label_dir / f"{stem}.txt").write_text(text, encoding="utf-8")
            classes = {int(line.split()[0]) for line in text.splitlines() if line.strip()}
            (with_keys if LEFTINCAR_KEYS_CLASS in classes else without_keys).append(stem)
        negatives = sorted(random.Random(args.seed).sample(
            without_keys, min(len(without_keys), round(len(with_keys) * args.negative_ratio))))
        selected = [*with_keys, *negatives]
        log(f"{len(with_keys)} images with keys, {len(negatives)} key-free backgrounds selected")
        for index, stem in enumerate(selected, 1):
            name = Path(images[stem]).name
            if not (image_dir / name).exists():
                temporary = image_dir / (name + ".part")
                temporary.write_bytes(archive.read(images[stem]))
                temporary.replace(image_dir / name)
            if index % 25 == 0:
                log(f"LeftInCAR images {index}/{len(selected)}")
    (LEFTINCAR_DIR / "selection.json").write_text(json.dumps(
        {"with_keys": with_keys, "negatives": negatives, "zip_sha256": LEFTINCAR_SHA256,
         "image_names": {s: Path(images[s]).name for s in selected}}, indent=2), encoding="utf-8")


# --------------------------------------------------------------------------- build
def yolo_line(x: float, y: float, w: float, h: float, width: int, height: int) -> str:
    x, y = max(0.0, x), max(0.0, y)
    w, h = min(w, width - x), min(h, height - y)
    return f"0 {(x + w / 2) / width:.6f} {(y + h / 2) / height:.6f} {w / width:.6f} {h / height:.6f}"


ZOOM_SIZE = 1280  # side of generated zoom images
ZOOM_MAX_FACTOR = 8  # same maximum enlargement as the crop/SR pipeline (x2/x4/x8)


def zoom_crops(image: np.ndarray, boxes: list[tuple[float, float, float, float]], count: int,
               rng: random.Random, sr) -> list[tuple[np.ndarray, list[str], dict]]:
    """Square crops around keys, resized so the key fills 20-80% of the side.

    Enlargement uses bicubic or FSRCNN, as the crop/SR pipeline would, never
    beyond x8. Crops that cut a key partially (10-60% visible) are rejected so
    every visible key keeps a correct box.
    """
    height, width = image.shape[:2]
    results = []
    for _ in range(count):
        for _attempt in range(20):
            target = rng.choice(boxes)
            target_w, target_h = target[2] - target[0], target[3] - target[1]
            side = max(target_w, target_h) / rng.uniform(0.2, 0.8)
            side = int(round(min(max(side, ZOOM_SIZE / ZOOM_MAX_FACTOR), width, height)))
            x0 = (target[0] + target[2]) / 2 - side / 2 + rng.uniform(-0.5, 0.5) * max(0.0, side - target_w)
            y0 = (target[1] + target[3]) / 2 - side / 2 + rng.uniform(-0.5, 0.5) * max(0.0, side - target_h)
            x0 = int(min(max(0.0, x0), width - side))
            y0 = int(min(max(0.0, y0), height - side))
            lines, partial = [], False
            for x1, y1, x2, y2 in boxes:
                cx1, cy1 = max(x1, x0), max(y1, y0)
                cx2, cy2 = min(x2, x0 + side), min(y2, y0 + side)
                if cx2 <= cx1 or cy2 <= cy1:
                    continue
                visible = (cx2 - cx1) * (cy2 - cy1) / max(1e-6, (x2 - x1) * (y2 - y1))
                if visible < 0.6:
                    partial = partial or visible > 0.1
                    continue
                lines.append(f"0 {((cx1 + cx2) / 2 - x0) / side:.6f} {((cy1 + cy2) / 2 - y0) / side:.6f} "
                             f"{(cx2 - cx1) / side:.6f} {(cy2 - cy1) / side:.6f}")
            if partial or not lines:
                continue
            region = image[y0:y0 + side, x0:x0 + side]
            factor = ZOOM_SIZE / side
            if factor <= 1:
                method, output = "area", cv2.resize(region, (ZOOM_SIZE, ZOOM_SIZE), interpolation=cv2.INTER_AREA)
            elif rng.random() < 0.5:
                method, output = "bicubic", cv2.resize(region, (ZOOM_SIZE, ZOOM_SIZE), interpolation=cv2.INTER_CUBIC)
            else:
                scale = next(s for s in (2, 4, 8) if s >= min(factor, 8))
                enhanced = sr.enhance(region, scale=scale)
                interpolation = cv2.INTER_AREA if enhanced.shape[0] >= ZOOM_SIZE else cv2.INTER_CUBIC
                method, output = f"fsrcnn_x{scale}", cv2.resize(enhanced, (ZOOM_SIZE, ZOOM_SIZE), interpolation=interpolation)
            results.append((output, lines, {"method": method, "factor": round(factor, 3),
                                            "crop": [x0, y0, side], "key_fraction": round(max(target_w, target_h) / side, 3)}))
            break
    return results


def add_zoom_images(output: Path, items: list[dict], split: str, target_split: str, count: int,
                    seed: int) -> dict:
    from small_object_lab.super_resolution import FSRCNNSuperResolution

    sr, rng = FSRCNNSuperResolution(), random.Random(seed)
    (output / "images" / target_split).mkdir(parents=True, exist_ok=True)
    (output / "labels" / target_split).mkdir(parents=True, exist_ok=True)
    made, methods, records = 0, {}, []
    for item in [i for i in items if i["split"] == split and i["lines"]]:
        image = cv2.imdecode(np.fromfile(item["image"], dtype=np.uint8), cv2.IMREAD_COLOR)
        height, width = image.shape[:2]
        boxes = []
        for line in item["lines"]:
            _, cx, cy, bw, bh = map(float, line.split())
            boxes.append(((cx - bw / 2) * width, (cy - bh / 2) * height, (cx + bw / 2) * width, (cy + bh / 2) * height))
        for index, (crop, lines, meta) in enumerate(zoom_crops(image, boxes, count, rng, sr)):
            stem = f"{item['stem']}_zoom{index}"
            ok, encoded = cv2.imencode(".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
            if not ok:
                raise OSError(f"JPEG encoding failed for {stem}")
            encoded.tofile(str(output / "images" / target_split / f"{stem}.jpg"))
            (output / "labels" / target_split / f"{stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
            made += 1
            methods[meta["method"].split("_")[0]] = methods.get(meta["method"].split("_")[0], 0) + 1
            records.append({"file": f"images/{target_split}/{stem}.jpg", "from": item["stem"], **meta})
    return {"images": made, "methods": methods, "records": records}


def build(args: argparse.Namespace) -> None:
    if args.output.exists():
        raise SystemExit(f"Output already exists: {args.output}")
    rng = random.Random(args.seed)
    items: list[dict] = []  # {source, image, lines, role}

    annotation = json.loads((O365_DIR / "annotations" / "zhiyuan_objv2_val.json").read_text(encoding="utf-8"))
    selection = o365_selection(annotation, args.negative_ratio, args.seed)
    images = {image["id"]: image for image in annotation["images"]}
    sample = next(iter(selection["boxes"].values()))[0]
    if abs(sample["bbox"][2] * sample["bbox"][3] - sample["area"]) > 1e-3 * max(1.0, sample["area"]):
        raise ValueError("Objects365 bbox is not [x, y, w, h]")
    for image_id in [*selection["positives"], *selection["negatives"]]:
        image = images[image_id]
        path = O365_DIR / "images" / Path(image["file_name"]).name
        lines = [yolo_line(*b["bbox"], image["width"], image["height"]) for b in selection["boxes"].get(image_id, [])]
        items.append({"source": "objects365_val", "image": path, "lines": lines, "license": image.get("license"),
                      "groups": sum(b.get("iscrowd", 0) for b in selection["boxes"].get(image_id, []))})

    left = json.loads((LEFTINCAR_DIR / "selection.json").read_text(encoding="utf-8"))
    for stem in [*left["with_keys"], *left["negatives"]]:
        text = (LEFTINCAR_DIR / "labels" / f"{stem}.txt").read_text(encoding="utf-8")
        lines = ["0 " + " ".join(line.split()[1:]) for line in text.splitlines()
                 if line.strip() and int(line.split()[0]) == LEFTINCAR_KEYS_CLASS]
        items.append({"source": "leftincar", "image": LEFTINCAR_DIR / "images" / left["image_names"][stem],
                      "lines": lines, "license": "CC BY 4.0", "groups": 0})

    missing = [str(i["image"]) for i in items if not i["image"].exists()]
    if missing:
        raise SystemExit(f"{len(missing)} images not fetched yet, e.g. {missing[:3]}")
    # Objects365 photos are independent, so they split randomly by image. LeftInCAR
    # frames come from continuous in-car captures; a random split would put
    # near-duplicates in both sets, so whole capture dates are held out instead.
    report = {}
    for source in ("objects365_val", "leftincar"):
        group = [i for i in items if i["source"] == source]
        rng.shuffle(group)
        if args.all_train:
            val, train = [], group
        elif source == "leftincar" and args.leftincar_val_dates:
            dates = set(args.leftincar_val_dates)
            val = [i for i in group if re.search(r"_(\d{4}-\d{2}-\d{2})_", i["image"].name).group(1) in dates]
            train = [i for i in group if i not in val]
        else:
            cut = round(len(group) * args.val_fraction)
            val, train = group[:cut], group[cut:]
        for split, members in (("val", val), ("train", train)):
            for item in members:
                item["split"] = split
            report[f"{source}/{split}"] = {"images": len(members), "with_keys": sum(bool(m["lines"]) for m in members),
                                           "boxes": sum(len(m["lines"]) for m in members),
                                           "group_boxes": sum(m["groups"] for m in members)}
    for index, item in enumerate(items):
        stem = f"{'o365' if item['source'] == 'objects365_val' else 'lic'}_{index:05d}"
        for kind in ("images", "labels"):
            (args.output / kind / item["split"]).mkdir(parents=True, exist_ok=True)
        shutil.copyfile(item["image"], args.output / "images" / item["split"] / f"{stem}{item['image'].suffix.lower()}")
        (args.output / "labels" / item["split"] / f"{stem}.txt").write_text(
            "\n".join(item["lines"]) + ("\n" if item["lines"] else ""), encoding="utf-8")
        item["stem"] = stem
    # Large keys: add zoomed crops (as the crop/SR path produces) to train, and a
    # separately generated zoomed copy of the held-out val images for evaluation.
    zoom = {}
    if args.zoom_per_image:
        zoom["train"] = add_zoom_images(args.output, items, "train", "train", args.zoom_per_image, args.seed + 1)
    if args.zoom_per_image and not args.all_train:
        zoom["val_zoom"] = add_zoom_images(args.output, items, "val", "val_zoom", args.zoom_per_image, args.seed + 2)
    for name, result in zoom.items():
        report[f"zoom/{name}"] = {k: v for k, v in result.items() if k != "records"}
    # Ultralytics requires a val set; with --all-train it only monitors training fit,
    # so train for a fixed number of epochs and use last.pt, not best.pt.
    if args.all_train:
        val_dir = "images/train"
    elif "val_zoom" in zoom:
        val_dir = "[images/val, images/val_zoom]"
    else:
        val_dir = "images/val"
    (args.output / "data.yaml").write_text(
        f"path: {args.output.resolve().as_posix()}\ntrain: images/train\nval: {val_dir}\nnames:\n  0: key\n",
        encoding="utf-8")
    manifest = {
        "schema_version": 1, "created_at": datetime.now(timezone.utc).isoformat(),
        "sources": {
            "objects365_val": {"annotation": "zhiyuan_objv2_val.json",
                               "annotation_sha256": hashlib.sha256((O365_DIR / "annotations" / "zhiyuan_objv2_val.json").read_bytes()).hexdigest(),
                               "terms": "Objects365: academic purposes only; images are not redistributable",
                               "excluded_fake_key_images": selection["excluded_fake"],
                               "note": "iscrowd boxes (groups such as key bunches) kept as ordinary boxes"},
            "leftincar": {"doi": "10.17632/dd4hw3r6dh.1", "license": "CC BY 4.0", "zip_sha256": left["zip_sha256"],
                          "class": "keys (7) -> key (0); other classes left unlabeled"},
        },
        "negative_ratio": args.negative_ratio, "val_fraction": args.val_fraction, "seed": args.seed,
        "leftincar_val_dates": args.leftincar_val_dates, "all_train": args.all_train,
        "zoom": {"per_image": args.zoom_per_image, "size": ZOOM_SIZE, "max_factor": ZOOM_MAX_FACTOR,
                 "key_fraction_range": [0.2, 0.8], "methods": "area / bicubic / FSRCNN x2-x8",
                 "files": {name: result["records"] for name, result in zoom.items()}},
        "splits": report,
        "files": [{"file": f"images/{i['split']}/{i['stem']}{i['image'].suffix.lower()}", "source": i["source"],
                   "original": i["image"].name, "license": i["license"], "boxes": len(i["lines"])} for i in items],
    }
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("fetch-objects365", "fetch-leftincar", "build"):
        command = sub.add_parser(name)
        command.add_argument("--negative-ratio", type=float, default=0.1,
                             help="Key-free background images per key image")
        command.add_argument("--seed", type=int, default=0)
        if name == "fetch-objects365":
            command.add_argument("--workers", type=int, default=8, help="Patch archives streamed in parallel")
        if name == "build":
            command.add_argument("--output", type=Path, required=True, help="New dataset directory")
            command.add_argument("--val-fraction", type=float, default=0.15)
            command.add_argument("--leftincar-val-dates", nargs="*", default=["2024-09-20"],
                                 help="LeftInCAR capture dates held out for validation (empty: random split)")
            command.add_argument("--all-train", action="store_true",
                                 help="Put every image in train (final model; evaluate on separate footage)")
            command.add_argument("--zoom-per-image", type=int, default=0,
                                 help="Zoomed key crops added per key image (large/upscaled keys)")
    args = parser.parse_args()
    {"fetch-objects365": fetch_objects365, "fetch-leftincar": fetch_leftincar, "build": build}[args.command](args)


if __name__ == "__main__":
    main()
