from pathlib import Path
import argparse

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Retrain the maritime ferry detector with more epochs and augmentation")
    parser.add_argument("--epochs", type=int, default=80, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--device", default="cpu", help="Training device, e.g. cpu or 0")
    parser.add_argument("--model", default=None, help="Optional path to a starting model (best.pt or yolov8m.pt)")
    parser.add_argument("--name", default="train_v3_more_data", help="Training run name")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent
    data_path = root / "dataset" / "data.yaml"

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset config not found: {data_path}")

    default_previous_model = root / "runs" / "detect" / "train_v2_151img" / "weights" / "best.pt"
    if args.model is None and default_previous_model.exists():
        model_path = str(default_previous_model)
    else:
        model_path = args.model or "yolov8m.pt"

    print(f"Starting training from: {model_path}")
    print(f"Dataset: {data_path}")

    model = YOLO(model_path)
    model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(root / "runs" / "detect"),
        name=args.name,
        pretrained=True,
        patience=50,
        single_cls=True,
        cache=False,
        amp=True,
        close_mosaic=10,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        erasing=0.4,
        mosaic=1.0,
        mixup=0.0,
        copy_paste=0.0,
    )

    export_path = model.export(format="openvino", imgsz=args.imgsz, half=False)
    print(f"Training finished. Exported model: {export_path}")


if __name__ == "__main__":
    main()
