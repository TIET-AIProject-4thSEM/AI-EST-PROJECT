"""
TATA STEEL — NEU-DET Binary Dataset Preparation (Final)
=======================================================

What this script does
---------------------
Creates a binary dataset for crack detection from NEU-DET XML annotations.

Important design choice
-----------------------
Because you do NOT have a separate folder of clean steel images, this script
keeps everything inside NEU-DET but fixes the biggest bias in the earlier
version:

1) CRACK samples are local patches cropped around defect bounding boxes.
2) NO_CRACK samples are local patches cropped from safe zones outside the
   defect box.
3) Dataset splitting happens at the *source image* level, so patches derived
   from the same original image never leak across train/val/test.

That means the model learns patch-level crack detection instead of a trivial
"full image vs crop" shortcut, while still staying within the data you have.

Expected input structure
------------------------
NEU-DET root:

    data/NEU-DET/
      train/
        images/
        annotations/
      validation/
        images/
        annotations/

Output structure
----------------
    data/steel_binary/
      train/CRACK/
      train/NO_CRACK/
      val/CRACK/
      val/NO_CRACK/
      test/CRACK/
      test/NO_CRACK/
      dataset_info.json
      sample_grid.png
      bbox_crop_illustration.png

Usage
-----
    python prepare_data_final.py --neu_dir data/NEU-DET --output_dir data/steel_binary

Notes
-----
- This script does not require any external NORMAL image folder.
- It is deliberately conservative: if no safe zone exists for an image, that
  image still contributes CRACK samples, but not NO_CRACK.
- The patch extraction is intentionally simple and robust for a college project.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import shutil
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [prepare] %(levelname)s  %(message)s",
)
logger = logging.getLogger("prepare")

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
IMG_SIZE = 224
MIN_SAFE_PX = 48
PAD = 4
DEFAULT_CRACK_PADDING = 10
DEFAULT_NOCRACK_PER_IMAGE = 2
DEFAULT_CRACK_PER_IMAGE = 1

# NEU-DET XML label variants → canonical label names
XML_LABEL_MAP = {
    "crazing": "crazing",
    "inclusion": "inclusion",
    "patches": "patches",
    "pitted_surface": "pitted_surface",
    "pitted surface": "pitted_surface",
    "rolled-in_scale": "rolled-in_scale",
    "rolled_in_scale": "rolled-in_scale",
    "rolled-in scale": "rolled-in_scale",
    "scratches": "scratches",
    "scratch": "scratches",
}

# All defect labels are considered CRACK in the binary formulation.
DEFECT_CLASSES = {
    "crazing",
    "inclusion",
    "patches",
    "pitted_surface",
    "rolled-in_scale",
    "scratches",
}


# -----------------------------------------------------------------------------
# Data structures
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class Box:
    label: str
    xmin: int
    ymin: int
    xmax: int
    ymax: int


@dataclass(frozen=True)
class Sample:
    source_id: str
    source_image: str
    label: int  # 1 = CRACK, 0 = NO_CRACK
    path: str
    split: str = ""


@dataclass(frozen=True)
class ImageRecord:
    """All samples derived from one source image."""

    source_id: str
    source_image: str
    crack_samples: Tuple[Sample, ...]
    no_crack_samples: Tuple[Sample, ...]


# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------
def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def clear_dir(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in IMG_EXTS


def find_image(stem: str, img_dir: Path) -> Optional[Path]:
    """Locate an image by stem inside img_dir or one nested subfolder."""
    for ext in IMG_EXTS:
        p = img_dir / f"{stem}{ext}"
        if p.exists():
            return p

    if not img_dir.exists():
        return None

    for sub in img_dir.iterdir():
        if not sub.is_dir():
            continue
        for ext in IMG_EXTS:
            p = sub / f"{stem}{ext}"
            if p.exists():
                return p
    return None


# -----------------------------------------------------------------------------
# XML parsing
# -----------------------------------------------------------------------------
def parse_xml(xml_path: Path) -> Optional[Dict]:
    """Parse one PASCAL-VOC XML file.

    Returns
    -------
    dict or None
        {
          'filename': str,
          'img_w': int,
          'img_h': int,
          'boxes': list[Box]
        }
    """
    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError:
        logger.warning(f"Skipping malformed XML: {xml_path}")
        return None

    filename = (root.findtext("filename") or "").strip()
    size_el = root.find("size")
    img_w = int(size_el.findtext("width", "200")) if size_el is not None else 200
    img_h = int(size_el.findtext("height", "200")) if size_el is not None else 200

    boxes: List[Box] = []
    for obj in root.findall("object"):
        raw = (obj.findtext("name") or "").strip().lower()
        label = XML_LABEL_MAP.get(raw)
        if label is None:
            continue

        bb = obj.find("bndbox")
        if bb is None:
            continue

        xmin = max(0, int(float(bb.findtext("xmin", "0"))))
        ymin = max(0, int(float(bb.findtext("ymin", "0"))))
        xmax = min(img_w, int(float(bb.findtext("xmax", str(img_w)))))
        ymax = min(img_h, int(float(bb.findtext("ymax", str(img_h)))))

        if xmax > xmin and ymax > ymin:
            boxes.append(Box(label, xmin, ymin, xmax, ymax))

    if not boxes:
        return None

    return {
        "filename": filename,
        "img_w": img_w,
        "img_h": img_h,
        "boxes": boxes,
    }


# -----------------------------------------------------------------------------
# Crop logic
# -----------------------------------------------------------------------------
def union_box(boxes: Sequence[Box]) -> Tuple[int, int, int, int]:
    xmin = min(b.xmin for b in boxes)
    ymin = min(b.ymin for b in boxes)
    xmax = max(b.xmax for b in boxes)
    ymax = max(b.ymax for b in boxes)
    return xmin, ymin, xmax, ymax


def get_safe_zones(
    xmin: int,
    ymin: int,
    xmax: int,
    ymax: int,
    img_w: int,
    img_h: int,
) -> List[Tuple[int, int, int, int]]:
    """Return rectangular regions fully outside the defect bbox.

    Candidate zones are strips around the bbox:
      - top
      - bottom
      - left
      - right

    A small inward padding is applied first to stay safely outside the border.
    Only zones with both width and height above MIN_SAFE_PX are kept.
    """
    ex1 = xmin + PAD
    ey1 = ymin + PAD
    ex2 = xmax - PAD
    ey2 = ymax - PAD

    candidates = [
        (0, 0, img_w, ey1),       # top
        (0, ey2, img_w, img_h),   # bottom
        (0, 0, ex1, img_h),       # left
        (ex2, 0, img_w, img_h),   # right
    ]

    valid: List[Tuple[int, int, int, int]] = []
    for x1, y1, x2, y2 in candidates:
        w = x2 - x1
        h = y2 - y1
        if w >= MIN_SAFE_PX and h >= MIN_SAFE_PX:
            valid.append((x1, y1, x2, y2))
    return valid


def crop_and_resize(img: np.ndarray, rect: Tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = rect
    patch = img[y1:y2, x1:x2]
    if patch.size == 0:
        raise ValueError("Empty crop region encountered")
    return cv2.resize(patch, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LANCZOS4)


def crop_defect_patch(
    img: np.ndarray,
    bbox: Tuple[int, int, int, int],
    pad: int = DEFAULT_CRACK_PADDING,
) -> np.ndarray:
    """Crop a localized patch around a defect bbox."""
    xmin, ymin, xmax, ymax = bbox
    h, w = img.shape[:2]

    x1 = max(0, xmin - pad)
    y1 = max(0, ymin - pad)
    x2 = min(w, xmax + pad)
    y2 = min(h, ymax + pad)

    return crop_and_resize(img, (x1, y1, x2, y2))


def crop_safe_patch(
    img: np.ndarray,
    zone: Tuple[int, int, int, int],
    rng: np.random.RandomState,
) -> np.ndarray:
    """Crop a patch from a safe zone and resize it to the model input size."""
    x1, y1, x2, y2 = zone
    region = img[y1:y2, x1:x2]
    rh, rw = region.shape[:2]

    crop_h = min(rh, IMG_SIZE)
    crop_w = min(rw, IMG_SIZE)

    max_top = max(0, rh - crop_h)
    max_left = max(0, rw - crop_w)
    top = rng.randint(0, max_top + 1)
    left = rng.randint(0, max_left + 1)

    patch = region[top : top + crop_h, left : left + crop_w]
    if patch.shape[0] != IMG_SIZE or patch.shape[1] != IMG_SIZE:
        patch = cv2.resize(patch, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LANCZOS4)
    return patch


# -----------------------------------------------------------------------------
# Scan NEU-DET and create per-image records
# -----------------------------------------------------------------------------
def scan_neu_det(
    neu_dir: Path,
    rng: np.random.RandomState,
    n_crack_per_image: int = DEFAULT_CRACK_PER_IMAGE,
    n_no_crack_per_image: int = DEFAULT_NOCRACK_PER_IMAGE,
) -> List[ImageRecord]:
    """Scan NEU-DET and create patch samples per source image.

    Each source image may contribute:
      - one or more CRACK patches
      - one or more NO_CRACK patches

    The source image is preserved as a grouping key so that splitting can be
    done at image level, preventing patch leakage across train/val/test.
    """
    records: List[ImageRecord] = []

    total_xml = 0
    skipped_noimg = 0
    skipped_nozone = 0
    skipped_noann = 0
    crack_count = 0
    no_crack_count = 0

    for split in ["train", "validation"]:
        ann_dir = neu_dir / split / "annotations"
        img_dir = neu_dir / split / "images"

        if not ann_dir.exists():
            logger.warning(f"Annotations folder missing: {ann_dir}")
            continue

        xml_files = sorted(ann_dir.rglob("*.xml"))
        logger.info(f"[{split}] XML files: {len(xml_files)}")

        for xml_path in xml_files:
            total_xml += 1
            ann = parse_xml(xml_path)
            if ann is None:
                skipped_noann += 1
                continue

            stem = Path(ann["filename"]).stem if ann["filename"] else xml_path.stem
            img_path = find_image(stem, img_dir) or find_image(xml_path.stem, img_dir)
            if img_path is None:
                skipped_noimg += 1
                continue

            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                skipped_noimg += 1
                continue

            # Convert to RGB once; downstream code expects standard RGB convention.
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_h, img_w = img_rgb.shape[:2]

            boxes: List[Box] = ann["boxes"]
            if not boxes:
                skipped_noann += 1
                continue

            # One source image -> one record. If multiple boxes exist, we use the
            # union bbox for safe-zone logic and crop CRACK patches per box.
            source_id = img_path.stem

            crack_samples: List[Sample] = []
            no_crack_samples: List[Sample] = []

            # -------- CRACK patches --------
            # Crop around each defect bbox, capped by n_crack_per_image.
            # If there are multiple boxes, we use up to the first few.
            for i, box in enumerate(boxes[:n_crack_per_image]):
                patch = crop_defect_patch(
                    img_rgb,
                    (box.xmin, box.ymin, box.xmax, box.ymax),
                    pad=DEFAULT_CRACK_PADDING,
                )
                crack_samples.append(
                    Sample(
                        source_id=source_id,
                        source_image=str(img_path),
                        label=1,
                        path="",
                    )
                )
                crack_count += 1

            # -------- NO_CRACK patches --------
            xmin, ymin, xmax, ymax = union_box(boxes)
            zones = get_safe_zones(xmin, ymin, xmax, ymax, img_w, img_h)

            if zones:
                rng.shuffle(zones)
                saved = 0
                for zone in zones:
                    if saved >= n_no_crack_per_image:
                        break
                    _ = crop_safe_patch(img_rgb, zone, rng)
                    no_crack_samples.append(
                        Sample(
                            source_id=source_id,
                            source_image=str(img_path),
                            label=0,
                            path="",
                        )
                    )
                    no_crack_count += 1
                    saved += 1
            else:
                skipped_nozone += 1

            # Only keep the source record if it contributed at least one sample.
            if crack_samples or no_crack_samples:
                records.append(
                    ImageRecord(
                        source_id=source_id,
                        source_image=str(img_path),
                        crack_samples=tuple(crack_samples),
                        no_crack_samples=tuple(no_crack_samples),
                    )
                )

    logger.info(
        f"Scanned XML files: {total_xml} | CRACK samples: {crack_count} | NO_CRACK samples: {no_crack_count}"
    )
    logger.info(
        f"Skipped -> no annotation: {skipped_noann}, no image: {skipped_noimg}, no safe zone: {skipped_nozone}"
    )
    return records


# -----------------------------------------------------------------------------
# Split at source-image level
# -----------------------------------------------------------------------------
def split_image_records(
    records: Sequence[ImageRecord],
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Dict[str, List[ImageRecord]]:
    """Split ImageRecord items so patches from the same source image stay together."""
    if not records:
        raise ValueError("No records available to split")

    train_records, temp_records = train_test_split(
        list(records),
        test_size=val_ratio + test_ratio,
        random_state=seed,
        shuffle=True,
    )

    vf = val_ratio / (val_ratio + test_ratio)
    val_records, test_records = train_test_split(
        temp_records,
        test_size=1 - vf,
        random_state=seed,
        shuffle=True,
    )

    return {
        "train": list(train_records),
        "val": list(val_records),
        "test": list(test_records),
    }


# -----------------------------------------------------------------------------
# Materialize files to output folders
# -----------------------------------------------------------------------------
def write_split(
    image_records: Sequence[ImageRecord],
    output_dir: Path,
    split_name: str,
    rng: np.random.RandomState,
) -> Tuple[int, int]:
    """Write CRACK and NO_CRACK patches for a split.

    Returns
    -------
    (n_crack, n_no_crack)
    """
    crack_dir = output_dir / split_name / "CRACK"
    no_crack_dir = output_dir / split_name / "NO_CRACK"
    ensure_dir(crack_dir)
    ensure_dir(no_crack_dir)

    crack_written = 0
    no_crack_written = 0

    # We need the actual patch images. Re-read the source images and crop again.
    # This keeps the file format clean and avoids storing temp intermediate files.
    for rec in image_records:
        img_bgr = cv2.imread(rec.source_image)
        if img_bgr is None:
            logger.warning(f"Could not read image during write: {rec.source_image}")
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        h, w = img_rgb.shape[:2]

        # Parse source image annotations again by looking up the XML that matches.
        # Since source_image is preserved from NEU-DET, use its stem to recover the
        # annotation from the corresponding split folder.
        #
        # The split folder is not stored here, so we will locate the XML by stem.
        # That is reliable because NEU-DET filenames are unique within a split.
        #
        # To avoid expensive full rescans, this function is intentionally simple
        # and the dataset is small enough for a college project.

        stem = Path(rec.source_image).stem
        xml_candidates = list(Path(rec.source_image).parents[2].joinpath("annotations").rglob(f"{stem}.xml"))
        if not xml_candidates:
            # Fallback: if the above path logic is not suitable, this record will
            # be skipped. In practice NEU-DET paths usually match the standard layout.
            logger.warning(f"Could not locate XML for {rec.source_image}")
            continue

        ann = parse_xml(xml_candidates[0])
        if ann is None:
            continue

        boxes: List[Box] = ann["boxes"]
        if not boxes:
            continue

        # --- Write CRACK patches ---
        for _ in rec.crack_samples:
            # Use the first defect box if multiple exist; for NEU-DET this is
            # usually just one box per image.
            box = boxes[0]
            patch = crop_defect_patch(
                img_rgb,
                (box.xmin, box.ymin, box.xmax, box.ymax),
                pad=DEFAULT_CRACK_PADDING,
            )
            out_name = f"crack_{crack_written:06d}.jpg"
            out_path = crack_dir / out_name
            cv2.imwrite(str(out_path), cv2.cvtColor(patch, cv2.COLOR_RGB2BGR))
            crack_written += 1

        # --- Write NO_CRACK patches ---
        xmin, ymin, xmax, ymax = union_box(boxes)
        zones = get_safe_zones(xmin, ymin, xmax, ymax, w, h)
        if zones:
            rng.shuffle(zones)
            saved = 0
            for zone in zones:
                if saved >= len(rec.no_crack_samples):
                    break
                patch = crop_safe_patch(img_rgb, zone, rng)
                out_name = f"nocrack_{no_crack_written:06d}.jpg"
                out_path = no_crack_dir / out_name
                cv2.imwrite(str(out_path), cv2.cvtColor(patch, cv2.COLOR_RGB2BGR))
                no_crack_written += 1
                saved += 1

    logger.info(f"{split_name}/CRACK: {crack_written} files")
    logger.info(f"{split_name}/NO_CRACK: {no_crack_written} files")
    return crack_written, no_crack_written


# -----------------------------------------------------------------------------
# Visualizations
# -----------------------------------------------------------------------------
def save_sample_grid(output_dir: Path, n: int = 6) -> None:
    """Save a grid of the first few training samples."""
    fig, axes = plt.subplots(2, n, figsize=(n * 3, 7))
    for row, label in enumerate(["CRACK", "NO_CRACK"]):
        folder = output_dir / "train" / label
        imgs = sorted([p for p in folder.iterdir() if is_image_file(p)])[:n]
        for col in range(n):
            ax = axes[row][col]
            ax.axis("off")
            if col >= len(imgs):
                continue
            img = cv2.imread(str(imgs[col]))
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ax.imshow(img)
            if col == 0:
                ax.set_ylabel(
                    label,
                    fontsize=13,
                    fontweight="bold",
                    rotation=0,
                    labelpad=80,
                    va="center",
                )

    plt.suptitle(
        "NEU-DET Binary Dataset\nCRACK = defect patch | NO_CRACK = safe-zone patch",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(output_dir / "sample_grid.png", dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved sample grid")


def save_bbox_crop_illustration(neu_dir: Path, output_dir: Path) -> None:
    """Save a small figure showing defect bbox and safe-zone idea."""
    for split in ["train", "validation"]:
        ann_root = neu_dir / split / "annotations"
        img_root = neu_dir / split / "images"
        if not ann_root.exists():
            continue

        for xml_path in ann_root.rglob("*.xml"):
            ann = parse_xml(xml_path)
            if ann is None:
                continue
            stem = Path(ann["filename"]).stem if ann["filename"] else xml_path.stem
            img_path = find_image(stem, img_root) or find_image(xml_path.stem, img_root)
            if img_path is None:
                continue

            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                continue
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            h, w = img_rgb.shape[:2]
            box = ann["boxes"][0]
            xmin, ymin, xmax, ymax = box.xmin, box.ymin, box.xmax, box.ymax
            zones = get_safe_zones(xmin, ymin, xmax, ymax, w, h)
            if not zones:
                continue

            fig, axes = plt.subplots(1, 3, figsize=(12, 4))

            axes[0].imshow(img_rgb)
            axes[0].add_patch(
                plt.Rectangle(
                    (xmin, ymin), xmax - xmin, ymax - ymin,
                    linewidth=2.5, edgecolor="red", facecolor="none",
                )
            )
            axes[0].set_title("Original image\n(red = defect bbox)", fontsize=10)
            axes[0].axis("off")

            axes[1].imshow(img_rgb)
            axes[1].add_patch(
                plt.Rectangle(
                    (xmin, ymin), xmax - xmin, ymax - ymin,
                    linewidth=2, edgecolor="red", facecolor="red", alpha=0.25,
                )
            )
            for zx1, zy1, zx2, zy2 in zones:
                axes[1].add_patch(
                    plt.Rectangle(
                        (zx1, zy1), zx2 - zx1, zy2 - zy1,
                        linewidth=2, edgecolor="lime", facecolor="lime", alpha=0.25,
                    )
                )
            axes[1].set_title("Safe zones (green)\nfor NO_CRACK patches", fontsize=10)
            axes[1].axis("off")

            sample_patch = crop_safe_patch(img_rgb, zones[0], np.random.RandomState(42))
            axes[2].imshow(sample_patch)
            axes[2].set_title("NO_CRACK patch\n(from safe zone)", fontsize=10)
            axes[2].axis("off")

            plt.suptitle(
                f"BBox → safe-zone patch extraction ({box.label})",
                fontsize=12,
                fontweight="bold",
            )
            plt.tight_layout()
            plt.savefig(output_dir / "bbox_crop_illustration.png", dpi=150, bbox_inches="tight")
            plt.close()
            logger.info("Saved bbox-crop illustration")
            return


# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
def count_images(folder: Path) -> int:
    if not folder.exists():
        return 0
    return sum(1 for p in folder.iterdir() if is_image_file(p))


def print_summary(output_dir: Path) -> None:
    print("\n" + "=" * 58)
    print("  STEEL BINARY DATASET — SUMMARY")
    print("=" * 58)
    print(f"  {'Split':<8}  {'CRACK':>8}  {'NO_CRACK':>10}  {'Total':>8}")
    print("  " + "─" * 42)

    grand = 0
    for split in ["train", "val", "test"]:
        c = count_images(output_dir / split / "CRACK")
        nc = count_images(output_dir / split / "NO_CRACK")
        tot = c + nc
        grand += tot
        print(f"  {split:<8}  {c:>8}  {nc:>10}  {tot:>8}")

    print("  " + "─" * 42)
    print(f"  {'TOTAL':<8}  {'':>8}  {'':>10}  {grand:>8}")
    print("=" * 58)
    print(f"\n  Location: {output_dir.resolve()}")
    print("\n  Next step:")
    print(f"    python crack_detector.py --data_dir {output_dir}")
    print("=" * 58 + "\n")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare NEU-DET as a binary CRACK / NO_CRACK patch dataset",
    )
    parser.add_argument(
        "--neu_dir",
        type=str,
        default="data/NEU-DET",
        help="NEU-DET root containing train/ and validation/ subfolders",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/steel_binary",
        help="Where to write the prepared dataset",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.15,
        help="Validation split fraction",
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.15,
        help="Test split fraction",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--crack_per_image",
        type=int,
        default=1,
        help="How many CRACK patches to create per source image",
    )
    parser.add_argument(
        "--no_crack_per_image",
        type=int,
        default=2,
        help="How many NO_CRACK patches to create per source image",
    )
    args = parser.parse_args()

    neu_dir = Path(args.neu_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.RandomState(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    print("\n" + "=" * 58)
    print("  TATA STEEL — NEU-DET Binary Preparation")
    print("=" * 58 + "\n")

    if not neu_dir.exists():
        print(f"[ERROR] NEU-DET not found at: {neu_dir}")
        print("Make sure the dataset root contains train/ and validation/ folders.")
        return

    # Clear output before writing a fresh dataset.
    clear_dir(output_dir)

    # ------------------------------------------------------------------
    # Scan and build records
    # ------------------------------------------------------------------
    image_records = scan_neu_det(
        neu_dir=neu_dir,
        rng=rng,
        n_crack_per_image=args.crack_per_image,
        n_no_crack_per_image=args.no_crack_per_image,
    )

    if not image_records:
        print("[ERROR] No valid images found. Check your NEU-DET folder structure.")
        return

    # The split must happen on source images, not on individual patches.
    splits = split_image_records(
        records=image_records,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    # ------------------------------------------------------------------
    # Write output dataset
    # ------------------------------------------------------------------
    logger.info("Writing output dataset ...")
    summary_counts = {}
    for split_name, split_records in splits.items():
        summary_counts[split_name] = write_split(split_records, output_dir, split_name, rng)

    # ------------------------------------------------------------------
    # Metadata
    # ------------------------------------------------------------------
    meta = {
        "neu_dir": str(neu_dir),
        "output_dir": str(output_dir),
        "img_size": IMG_SIZE,
        "min_safe_px": MIN_SAFE_PX,
        "pad": PAD,
        "crack_padding": DEFAULT_CRACK_PADDING,
        "crack_per_image": args.crack_per_image,
        "no_crack_per_image": args.no_crack_per_image,
        "seed": args.seed,
        "split_strategy": "source-image level split to prevent patch leakage",
        "class_definition": {
            "CRACK": "localized defect patch around bbox",
            "NO_CRACK": "localized patch from safe zone outside bbox",
        },
        "counts": {
            split: {"CRACK": c, "NO_CRACK": nc} for split, (c, nc) in summary_counts.items()
        },
    }
    with open(output_dir / "dataset_info.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    # ------------------------------------------------------------------
    # Visuals
    # ------------------------------------------------------------------
    save_sample_grid(output_dir)
    save_bbox_crop_illustration(neu_dir, output_dir)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print_summary(output_dir)


if __name__ == "__main__":
    main()

