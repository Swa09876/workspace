import sys
import os
import argparse
import yaml

def load_config(path="config.yaml"):
    with open(path, "r", encoding="utf8") as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_path = cfg.get("model", {}).get("path")
    device = cfg.get("model", {}).get("device")

    if model_path is None:
        print("No model.path found in config.yaml")
        sys.exit(2)

    print(f"Model path (from config): {model_path}")
    if not os.path.exists(model_path):
        print(f"Warning: model file does not exist at {model_path}")
        print("If the path is remote or you expect ultralytics to auto-download, this may be fine.")

    try:
        from ultralytics import YOLO
    except Exception as e:
        print("Cannot import ultralytics. Install project dependencies first:")
        print("    pip install -r requirements.txt")
        raise

    print("Attempting to load model with ultralytics.YOLO() (this may take a moment)...")
    try:
        model = YOLO(model_path)
        print("Model loaded successfully.")
        try:
            model.info()
        except Exception:
            # model.info may print internally; if it errors, ignore
            pass
        names = getattr(model, 'names', None)
        if names:
            print(f"Model classes ({len(names)}): {list(names)[:10]}{('...' if len(names)>10 else '')}")
    except Exception as e:
        print("Failed to load model:", e)
        sys.exit(3)

if __name__ == '__main__':
    main()
