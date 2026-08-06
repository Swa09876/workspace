# YOLO Auto-Annotate + Roboflow Upload

A pipeline that uses a pre-trained YOLO model to automatically annotate your
images and then uploads the annotated images + labels to a Roboflow dataset in
parallel, so you can immediately generate new dataset versions from Roboflow.

## How it works

```text
You drop images into data/input/
            |
            v
[1] YOLO inference  ->  data/labels/*.txt  (YOLO format annotations)
[2] images copied   ->  data/images/*      (paired with the .txt files)
[3] visualizations  ->  data/annotated/*   (boxes drawn, optional)
            |
            v
[4] Parallel upload -> Roboflow project (image + label pairs)
            |
            v
[5] Create a new dataset version in Roboflow from the "auto_annotated" batch
```

## Folder structure

```text
.
├── auto_annotate_and_upload.py   # main script (annotate + upload)
├── config.yaml                   # all settings live here
├── requirements.txt              # Python dependencies
├── README.md
└── data/
    ├── input/        # PUT YOUR RAW IMAGES HERE
    ├── labels/       # generated YOLO .txt label files (auto-created)
    ├── images/       # copies of images paired with labels (auto-created)
    ├── annotated/    # visualizations with boxes drawn (auto-created)
    └── log/          # run logs + manifest (auto-created)
```

## Installation

```bash
pip install --break-system-packages -r requirements.txt
```

Optional (faster inference on CPU-only machines, smaller download):

```bash
pip install --break-system-packages torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

## Setup

1. Install the dependencies (above).
2. Edit `config.yaml`:
   - `model.path` — which pre-trained YOLO to use (`yolov8n.pt`, `yolov8s.pt`,
     `yolov11n.pt`, or a path to your own `best.pt`).
   - `roboflow.workspace` and `roboflow.project` — your Roboflow IDs.
   - Set the API key via environment variable (recommended, keeps it out of git):

     ```bash
     export ROBOFLOW_API_KEY=your_roboflow_api_key
     ```

     or paste it into `roboflow.api_key` in `config.yaml`.
3. Drop your images into `data/input/`.

## Usage

Annotate locally only (no upload) — good for testing:

```bash
python auto_annotate_and_upload.py --config config.yaml --dry-run
```

Full run (annotate + upload to Roboflow in parallel):

```bash
python auto_annotate_and_upload.py --config config.yaml
```

Override the model from the command line:

```bash
python auto_annotate_and_upload.py --config config.yaml --model yolov11s.pt
```

## After the run

1. Check `data/log/manifest_*.json` and the `.log` file for a summary.
2. In Roboflow, go to your project -> Uploads (or the "Datasets" tab) and you
   will see the uploaded images under the `auto_annotated` batch.
3. Generate a **new version** of the dataset in Roboflow (Add/Edit version),
   optionally clean up low-confidence boxes, then export the version for
   training your fine-tuned model.

## Notes / tips

- Images with **zero detections** are skipped by default
  (`roboflow.skip_empty: true`). Set it to `false` to upload them anyway.
- Increase `roboflow.workers` for faster uploads, lower it if you hit Roboflow
  rate limits.
- The Roboflow class names are taken automatically from the model's class
  list, so annotations are matched to existing classes. If your project uses
  different class names, add/rename classes in Roboflow *before* uploading.
- Re-running processes every image in `data/input` again. Move processed images
  out of `data/input` (e.g. into a `data/done/` folder) if you do not want to
  re-annotate them.
