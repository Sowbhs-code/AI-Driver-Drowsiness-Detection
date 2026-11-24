"""
train_yolo_phone.py
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Fine-tune YOLOv8n on a phone-detection dataset.

Dataset  : Phone Detection Dataset (Roboflow)
           https://universe.roboflow.com/roboflow-100/cell-phones-detection
           or download via:  pip install roboflow

Usage    : python training/train_yolo_phone.py
Output   : yolov8_model/phone_detector_best.pt
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

import os
import yaml

# ── Config ─────────────────────────────────────────────────────────────────────
CFG = {
    "base_model"  : "yolov8n.pt",           # pretrained base weights
    "output_dir"  : "yolov8_model",
    "data_yaml"   : "dataset/phone_data/data.yaml",
    "epochs"      : 50,
    "img_size"    : 640,
    "batch_size"  : 16,
    "device"      : "0",                    # GPU id or 'cpu'
    "project"     : "yolov8_model/runs",
    "name"        : "phone_detector",
    "patience"    : 10,                     # early stopping patience
}

os.makedirs(CFG["output_dir"], exist_ok=True)
os.makedirs("dataset/phone_data", exist_ok=True)


def create_data_yaml():
    """Write a sample data.yaml — replace paths with your actual dataset."""
    data = {
        "path"  : os.path.abspath("dataset/phone_data"),
        "train" : "images/train",
        "val"   : "images/val",
        "nc"    : 1,
        "names" : ["cell phone"],
    }
    with open(CFG["data_yaml"], "w") as f:
        yaml.dump(data, f, default_flow_style=False)
    print(f"[INFO] data.yaml created at {CFG['data_yaml']}")
    print("[INFO] Place your dataset images in:")
    print(f"         dataset/phone_data/images/train/")
    print(f"         dataset/phone_data/images/val/")
    print(f"         dataset/phone_data/labels/train/   (YOLO .txt format)")
    print(f"         dataset/phone_data/labels/val/")


def train():
    try:
        from ultralytics import YOLO
    except ImportError:
        print("[ERROR] ultralytics not installed. Run: pip install ultralytics")
        return

    if not os.path.exists(CFG["data_yaml"]):
        create_data_yaml()
        print("\n[WARN] data.yaml was just created with placeholder paths.")
        print("       Add your dataset images and re-run this script.")
        return

    print(f"[INFO] Starting YOLOv8 fine-tuning on phone detection dataset...")
    print(f"       Base model : {CFG['base_model']}")
    print(f"       Epochs     : {CFG['epochs']}")
    print(f"       Image size : {CFG['img_size']}")

    model = YOLO(CFG["base_model"])

    results = model.train(
        data      = CFG["data_yaml"],
        epochs    = CFG["epochs"],
        imgsz     = CFG["img_size"],
        batch     = CFG["batch_size"],
        device    = CFG["device"],
        project   = CFG["project"],
        name      = CFG["name"],
        patience  = CFG["patience"],
        save      = True,
        plots     = True,
        verbose   = True,
    )

    # Copy best weights to model output directory
    best_pt = os.path.join(CFG["project"], CFG["name"], "weights", "best.pt")
    if os.path.exists(best_pt):
        import shutil
        dest = os.path.join(CFG["output_dir"], "phone_detector_best.pt")
        shutil.copy2(best_pt, dest)
        print(f"\n[DONE] Best weights saved to: {dest}")

    # Print validation metrics
    metrics = model.val()
    print("\n── Validation Metrics ──────────────────────────────────")
    print(f"  mAP50      : {metrics.box.map50:.4f}")
    print(f"  mAP50-95   : {metrics.box.map:.4f}")
    print(f"  Precision  : {metrics.box.mp:.4f}")
    print(f"  Recall     : {metrics.box.mr:.4f}")


def download_roboflow_dataset(api_key: str, workspace: str,
                               project: str, version: int = 1):
    """Helper: download dataset from Roboflow (requires roboflow package)."""
    try:
        from roboflow import Roboflow
        rf      = Roboflow(api_key=api_key)
        proj    = rf.workspace(workspace).project(project)
        dataset = proj.version(version).download("yolov8", location="dataset/phone_data")
        print(f"[INFO] Dataset downloaded to dataset/phone_data")
        return dataset
    except ImportError:
        print("[ERROR] roboflow not installed. Run: pip install roboflow")
    except Exception as e:
        print(f"[ERROR] Download failed: {e}")


if __name__ == "__main__":
    train()
