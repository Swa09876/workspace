#!/usr/bin/env python3
"""
Auto-annotate images with a pre-trained YOLO model and upload the
annotated dataset to a Roboflow project.

Pipeline:
  1. Scan images in data/input
  2. Run inference with the configured YOLO model
  3. Write YOLO-format label files (*.txt) into data/labels
  4. Copy the matching images into data/images
  5. Optionally render visualizations into data/annotated
  6. Upload image + label pairs to Roboflow in parallel

Usage:
  python auto_annotate_and_upload.py --config config.yaml
  python auto_annotate_and_upload.py --config config.yaml --dry-run
  python auto_annotate_and_upload.py --config config.yaml --model yolov11n.pt
"""

import argparse
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import yaml

import traceback
import base64
import tempfile

LOG = logging.getLogger("auto_annotate")

# A deterministic palette so a class keeps the same box color across images.
_COLOR_PALETTE = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255),
    (255, 0, 255), (255, 128, 0), (128, 255, 0), (0, 128, 255), (255, 0, 128),
    (128, 0, 255), (0, 255, 128), (255, 128, 128), (128, 255, 128),
    (128, 128, 255), (255, 255, 128), (255, 128, 255), (128, 255, 255),
    (64, 0, 0), (0, 64, 0), (0, 0, 64), (192, 64, 0), (0, 192, 64),
    (64, 0, 192), (192, 0, 64), (64, 192, 0), (0, 64, 192), (192, 64, 64),
]


def load_config(config_path: str) -> dict:
    path = Path(config_path)
    if not path.exists():
        raise SystemExit(f"Config file not found: {path}")
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if not isinstance(cfg, dict):
        raise SystemExit(f"Invalid config file: {config_path}")
    return cfg


def setup_logging(log_dir: Path) -> Path:
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(fmt)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(fmt)

    LOG.setLevel(logging.INFO)
    LOG.addHandler(file_handler)
    LOG.addHandler(console_handler)
    LOG.propagate = False
    return log_file


def ensure_dirs(cfg: dict) -> dict:
    """Create output directories and return their Path objects."""
    in_cfg = cfg["input"]
    out_cfg = cfg["output"]
    dirs = {
        "input": Path(in_cfg["images_dir"]),
        "labels": Path(out_cfg["labels_dir"]),
        "images": Path(out_cfg["images_dir"]),
        "annotated": Path(out_cfg["annotated_dir"]),
        "log": Path(out_cfg["log_dir"]),
    }
    dirs["input"].mkdir(parents=True, exist_ok=True)
    for key in ("labels", "images", "annotated", "log"):
        dirs[key].mkdir(parents=True, exist_ok=True)
    return dirs


def discover_images(images_dir: Path, extensions: list) -> list:
    if not images_dir.exists():
        raise SystemExit(f"Input images directory does not exist: {images_dir}")
    exts = tuple(ext.lower() for ext in extensions)
    images = sorted(
        p for p in images_dir.rglob("*")
        if p.is_file() and p.suffix.lower() in exts
    )
    if not images:
        LOG.warning("No images found in %s. Drop images there and re-run.", images_dir)
    return images


def resolve_api_key(cfg_key: str) -> str:
    key = (cfg_key or "").strip()
    env_key = os.environ.get("ROBOFLOW_API_KEY", "").strip()
    if env_key:
        return env_key
    if key:
        return key
    return ""


def resolve_batch_name(cfg: dict) -> str:
    raw_name = str(cfg["roboflow"].get("batch_name", "")).strip() or "auto_annotated"
    if raw_name == "auto_annotated":
        raw_name = f"{raw_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    return raw_name


def render_visualization(image_bgr, results, names: dict, out_path: Path) -> None:
    """Draw segmentation masks onto a copy of the image."""
    annotated = image_bgr.copy()

    # Prefer segmentation masks if available
    if hasattr(results, "masks") and results.masks is not None:
        masks = None
        try:
            masks = results.masks.data.cpu().numpy()
        except Exception:
            try:
                masks = results.masks.cpu().numpy()
            except Exception:
                masks = None

        if masks is not None and len(masks) > 0:
            cls_ids = results.boxes.cls.cpu().numpy().astype(int) if results.boxes is not None else [0] * len(masks)
            h, w = annotated.shape[:2]
            overlay = np.zeros((h, w, 3), dtype=np.uint8)
            for mask, cls_id in zip(masks, cls_ids):
                color = _COLOR_PALETTE[cls_id % len(_COLOR_PALETTE)]
                mh, mw = mask.shape[:2]
                if (mh, mw) != (h, w):
                    mask_uint8 = (mask.astype('uint8') * 255) if mask.dtype != np.uint8 else mask
                    mask_resized = cv2.resize(mask_uint8, (w, h), interpolation=cv2.INTER_NEAREST)
                    mask_bool = mask_resized.astype(bool)
                else:
                    mask_bool = mask.astype(bool) if mask.dtype != np.bool_ else mask
                overlay[mask_bool] = color
            # Blend mask overlay onto original image
            cv2.addWeighted(overlay, 0.5, annotated, 0.5, 0, annotated)

            # Put small class label tag near mask centroid
            try:
                confs = results.boxes.conf.cpu().numpy() if results.boxes is not None else [1.0] * len(masks)
                boxes = results.boxes.xyxy.cpu().numpy() if results.boxes is not None else []
                for idx, (conf, cls_id) in enumerate(zip(confs, cls_ids)):
                    if idx < len(boxes):
                        x1, y1 = int(boxes[idx][0]), int(boxes[idx][1])
                    else:
                        x1, y1 = 10, 10 + idx * 20
                    label = f"{names.get(int(cls_id), cls_id)} {conf:.2f}"
                    color = _COLOR_PALETTE[int(cls_id) % len(_COLOR_PALETTE)]
                    text_w = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0][0]
                    cv2.rectangle(annotated, (max(0, x1), max(0, y1 - 15)), (max(0, x1) + text_w + 6, max(0, y1)), color, -1)
                    cv2.putText(
                        annotated, label, (max(0, x1) + 3, max(0, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA,
                    )
            except Exception:
                pass
            cv2.imwrite(str(out_path), annotated)
            return

    # Fallback if masks not available
    if results is None or results.boxes is None or len(results.boxes) == 0:
        cv2.imwrite(str(out_path), image_bgr)
        return
    boxes = results.boxes.xyxy.cpu().numpy()
    confs = results.boxes.conf.cpu().numpy()
    cls_ids = results.boxes.cls.cpu().numpy().astype(int)
    for box, conf, cls_id in zip(boxes, confs, cls_ids):
        x1, y1, x2, y2 = (int(v) for v in box)
        color = _COLOR_PALETTE[cls_id % len(_COLOR_PALETTE)]
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        label = f"{names.get(cls_id, cls_id)} {conf:.2f}"
        text_w = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0]
        cv2.rectangle(annotated, (x1, y1 - 18), (x1 + text_w + 6, y1), color, -1)
        cv2.putText(
            annotated, label, (x1 + 3, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA,
        )
    cv2.imwrite(str(out_path), annotated)


def annotate_image(model, image_path: Path, cfg: dict, dirs: dict, names: dict) -> dict:
    """
    Run inference on one image and persist the YOLO label file (polygon mask format),
    image copy and (optional) visualization. Returns a record dict for the upload phase.
    """
    m_cfg = cfg["model"]

    results = model.predict(
        source=str(image_path),
        conf=m_cfg["conf_threshold"],
        iou=m_cfg["iou_threshold"],
        imgsz=m_cfg["image_size"],
        device=m_cfg.get("device"),
        verbose=False,
    )[0]

    detections = []
    mask_path = None
    stem = image_path.stem
    image_out = dirs["images"] / image_path.name
    label_path = dirs["labels"] / f"{stem}.txt"

    # Extract polygon segmentation masks for YOLO segmentation format
    if results.boxes is not None and len(results.boxes) > 0:
        cls_ids = results.boxes.cls.cpu().numpy().astype(int)
        confs = results.boxes.conf.cpu().numpy()

        if hasattr(results, "masks") and results.masks is not None and hasattr(results.masks, "xyn") and len(results.masks.xyn) > 0:
            polygons_xyn = results.masks.xyn
            for cls_id, conf, poly in zip(cls_ids, confs, polygons_xyn):
                if len(poly) >= 3:  # Polygon requires at least 3 points
                    poly_flat = poly.flatten().tolist()
                    detections.append({
                        "class_id": int(cls_id),
                        "conf": float(conf),
                        "polygon": [float(v) for v in poly_flat],
                    })
        else:
            # Fallback to bbox if masks unavailable
            xywhn = results.boxes.xywhn.cpu().numpy()
            for cls_id, conf, box in zip(cls_ids, confs, xywhn):
                detections.append({
                    "class_id": int(cls_id),
                    "conf": float(conf),
                    "xywhn": [float(v) for v in box],
                })

    with open(label_path, "w", encoding="utf-8") as fh:
        for det in detections:
            if "polygon" in det:
                poly_str = " ".join(f"{v:.6f}" for v in det["polygon"])
                fh.write(f"{det['class_id']} {poly_str}\n")
            elif "xywhn" in det:
                cx, cy, w, h = det["xywhn"]
                fh.write(f"{det['class_id']} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

    if image_path.resolve() != image_out.resolve():
        shutil_copy(image_path, image_out)

    if m_cfg.get("save_visualizations", True):
        annotated_out = dirs["annotated"] / f"{stem}.jpg"
        render_visualization(cv2.imread(str(image_path)), results, names, annotated_out)

    return {
        "image_path": image_out,
        "annotation_path": label_path,
        "mask_path": mask_path,
        "num_detections": len(detections),
        "class_ids": sorted({d["class_id"] for d in detections}) if detections else [],
    }


def shutil_copy(src: Path, dst: Path) -> None:
    import shutil
    shutil.copy2(str(src), str(dst))


def upload_image(project, record: dict, cfg: dict, class_map: dict) -> dict:
    """Upload a single image + its YOLO label to Roboflow. Returns status info."""
    r_cfg = cfg["roboflow"]
    # If the annotation is a binary mask (PNG), Roboflow SDK opens it in text mode
    # which causes UnicodeDecodeError. Convert PNG to a base64 data-URI text file
    # so the SDK can read it safely and the server receives the PNG bytes.
    ann_path = str(record["annotation_path"])
    temp_path = None
    try:
        if ann_path.lower().endswith('.png'):
            # read binary and write a temporary text file with data URI
            with open(ann_path, 'rb') as f:
                b = f.read()
            b64 = base64.b64encode(b).decode('ascii')
            data_uri = f"data:image/png;base64,{b64}"
            tf = tempfile.NamedTemporaryFile(delete=False, suffix='.txt', mode='w', encoding='utf-8')
            tf.write(data_uri)
            tf.close()
            temp_path = tf.name
            ann_to_send = temp_path
        else:
            ann_to_send = ann_path

        project.single_upload(
            image_path=str(record["image_path"]),
            annotation_path=str(ann_to_send),
            annotation_labelmap=class_map,
            split=r_cfg.get("split", "train"),
            num_retry_uploads=r_cfg.get("num_retries", 3),
            batch_name=r_cfg.get("batch_name", "auto_annotated"),
            annotation_overwrite=r_cfg.get("overwrite", True),
        )
        return {"image_path": record["image_path"], "status": "uploaded"}
    except Exception as exc:  # noqa: BLE001 - report every failure, keep going
        # Safely stringify exception to avoid encoding errors (e.g., 'charmap' decode failures)
        try:
            # If exception args contain bytes, decode them
            parts = []
            for a in getattr(exc, 'args', (str(exc),)):
                if isinstance(a, bytes):
                    parts.append(a.decode('utf-8', errors='replace'))
                else:
                    parts.append(str(a))
            err_text = " ".join(parts)
        except Exception:
            err_text = repr(exc)
        # Include traceback for diagnostics
        tb = traceback.format_exc()
        safe_error = f"{err_text} | traceback: {tb}"
        return {"image_path": record["image_path"], "status": "failed", "error": safe_error}
    finally:
        if temp_path is not None:
            try:
                os.remove(temp_path)
            except Exception:
                pass


def upload_all(project, records: list, cfg: dict, class_map: dict) -> dict:
    """Upload annotated pairs to Roboflow in parallel and report results."""
    workers = max(1, int(cfg["roboflow"].get("workers", 4)))
    results = {"uploaded": 0, "failed": 0, "failures": []}
    if not records:
        LOG.info("Nothing to upload.")
        return results

    LOG.info("Uploading %d image+label pairs to Roboflow (workers=%d) ...",
             len(records), workers)
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(upload_image, project, record, cfg, class_map): record
            for record in records
        }
        for future in as_completed(futures):
            info = future.result()
            if info["status"] == "uploaded":
                results["uploaded"] += 1
                LOG.info("[OK] %s", info["image_path"].name)
            else:
                results["failed"] += 1
                results["failures"].append(info)
                LOG.error("[FAIL] %s -> %s", info["image_path"].name, info["error"])
    return results


def write_manifest(dirs: dict, records: list, results: dict, cfg: dict,
                   class_map: dict) -> None:
    """Persist a human/machine readable summary of the run."""
    # Convert any Path objects inside results to strings so YAML dumping succeeds
    safe_results = {
        "uploaded": int(results.get("uploaded", 0)),
        "failed": int(results.get("failed", 0)),
        "failures": [],
    }
    for f in results.get("failures", []):
        safe_f = {}
        for k, v in f.items():
            if isinstance(v, (Path,)):
                safe_f[k] = str(v)
            else:
                try:
                    # attempt to convert common types to serializable forms
                    if isinstance(v, bytes):
                        safe_f[k] = v.decode("utf-8", errors="replace")
                    else:
                        safe_f[k] = v
                except Exception:
                    safe_f[k] = str(v)
        safe_results["failures"].append(safe_f)

    manifest = {
        "timestamp": datetime.now().isoformat(),
        "model": cfg["model"]["path"],
        "batch_name": cfg["roboflow"].get("batch_name"),
        "class_map": {str(k): v for k, v in class_map.items()},
        "records": [
            {
                "image": str(r["image_path"].name) if hasattr(r.get("image_path"), "name") else str(r.get("image_path")),
                "num_detections": int(r.get("num_detections", 0)),
                "class_ids": r.get("class_ids", []),
            }
            for r in records
        ],
        "upload": safe_results,
    }
    out_path = dirs["log"] / f"manifest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(manifest, fh, sort_keys=False, allow_unicode=True)
    LOG.info("Manifest written to %s", out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--dry-run", action="store_true",
                        help="Annotate locally but do NOT upload to Roboflow")
    parser.add_argument("--model", default=None,
                        help="Override the model path from config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.model:
        cfg["model"]["path"] = args.model

    dirs = ensure_dirs(cfg)
    log_file = setup_logging(dirs["log"])
    LOG.info("Log file: %s", log_file)
    LOG.info("Loading model: %s", cfg["model"]["path"])

    from ultralytics import YOLO
    model = YOLO(cfg["model"]["path"])
    names = model.names
    class_map = {int(k): v for k, v in names.items()}
    LOG.info("Model loaded. Classes: %s", ", ".join(class_map.values()))

    images = discover_images(dirs["input"], cfg["input"]["extensions"])
    if not images:
        LOG.warning("No input images found. Exiting.")
        return

    LOG.info("Annotating %d images...", len(images))
    records = []
    skipped = 0
    for i, img in enumerate(images, start=1):
        record = annotate_image(model, img, cfg, dirs, names)
        if record["num_detections"] == 0 and cfg["roboflow"].get("skip_empty", True):
            skipped += 1
            LOG.info("[%d/%d] %s -> 0 detections, skipped",
                     i, len(images), img.name)
            continue
        records.append(record)
        LOG.info("[%d/%d] %s -> %d detections",
                 i, len(images), img.name, record["num_detections"])

    if skipped:
        LOG.info("Skipped %d empty images.", skipped)
    LOG.info("Annotated %d images (labels in %s, images in %s).",
             len(records), dirs["labels"], dirs["images"])

    api_key = resolve_api_key(cfg["roboflow"].get("api_key", ""))
    upload_enabled = cfg["roboflow"].get("upload_enabled", True)
    batch_name = resolve_batch_name(cfg)
    cfg["roboflow"]["batch_name"] = batch_name

    LOG.info("Roboflow batch name: %s", batch_name)

    results = {"uploaded": 0, "failed": 0, "failures": []}
    if args.dry_run:
        LOG.info("Dry-run mode: upload skipped.")
    elif not upload_enabled:
        LOG.info("upload_enabled=false in config: upload skipped.")
    elif not api_key or not cfg["roboflow"].get("workspace") or not cfg["roboflow"].get("project"):
        LOG.error(
            "Roboflow is not configured. Set ROBOFLOW_API_KEY env var (or "
            "roboflow.api_key in config.yaml) plus roboflow.workspace and "
            "roboflow.project in config.yaml. Exiting without upload."
        )
    else:
        from roboflow import Roboflow
        rf = Roboflow(api_key=api_key)
        project = (
            rf.workspace(cfg["roboflow"]["workspace"])
              .project(cfg["roboflow"]["project"])
        )
        results = upload_all(project, records, cfg, class_map)

    write_manifest(dirs, records, results, cfg, class_map)
    LOG.info("Done. Uploaded=%d Failed=%d", results["uploaded"], results["failed"])
    if results["failed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()


