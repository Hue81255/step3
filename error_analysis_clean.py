
from pathlib import Path
from ultralytics import YOLO
import shutil
import csv

BASE = Path(r"C:\Users\lee\step3-defect")

DATA = Path(
    r"C:\Users\lee\step3-defect\test_clean_39"
)

MODEL = (
    BASE / "runs" / "segment" / "runs"
    / "step3-5" / "weights" / "best.pt"
)

IMAGE_DIR = DATA / "images" / "test"
LABEL_DIR = DATA / "labels" / "test"

OUTPUT = BASE / "error_cases_clean"

CLASSES = {
    0: "crack",
    1: "efflorescence",
    2: "spall_detach"
}

THRESHOLDS = {
    0: 0.2,
    1: 0.02,
    2: 0.2
}


def main():
    if not MODEL.exists():
        raise FileNotFoundError(MODEL)

    if not IMAGE_DIR.exists():
        raise FileNotFoundError(IMAGE_DIR)

    model = YOLO(str(MODEL))

    images = sorted(IMAGE_DIR.glob("*.jpg"))

    if len(images) != 39:
        raise RuntimeError(
            f"39장이 필요하지만 {len(images)}장이 발견됨"
        )

    records = []
    counts = {
        (cid, kind): 0
        for cid in CLASSES
        for kind in ("FP", "FN")
    }

    for image in images:
        label = LABEL_DIR / (image.stem + ".txt")

        if not label.exists():
            raise FileNotFoundError(label)

        gt_classes = set()

        for line in label.read_text(
            encoding="utf-8"
        ).splitlines():
            parts = line.split()
            if parts:
                gt_classes.add(int(parts[0]))

        result = model.predict(
            source=str(image),
            conf=0.001,
            imgsz=960,
            device=0,
            verbose=False
        )[0]

        pred_classes = set()

        if result.boxes is not None:
            for cls, confidence in zip(
                result.boxes.cls.tolist(),
                result.boxes.conf.tolist()
            ):
                cid = int(cls)

                if (
                    cid in THRESHOLDS
                    and confidence >= THRESHOLDS[cid]
                ):
                    pred_classes.add(cid)

        for cid, name in CLASSES.items():
            gt = cid in gt_classes
            pred = cid in pred_classes

            if pred and not gt:
                kind = "FP"
            elif gt and not pred:
                kind = "FN"
            else:
                continue

            counts[(cid, kind)] += 1

            folder = OUTPUT / name / kind
            folder.mkdir(parents=True, exist_ok=True)

            destination = folder / image.name
            shutil.copy2(image, destination)

            records.append({
                "class": name,
                "type": kind,
                "image": image.name,
                "threshold": THRESHOLDS[cid]
            })

    OUTPUT.mkdir(parents=True, exist_ok=True)

    csv_path = OUTPUT / "error_summary.csv"

    with csv_path.open(
        "w", newline="", encoding="utf-8-sig"
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "class", "type", "image", "threshold"
            ]
        )
        writer.writeheader()
        writer.writerows(records)

    print("\n===== FP/FN 오류 사례 =====")

    for cid, name in CLASSES.items():
        print(f"\n{name}")
        print("FP:", counts[(cid, "FP")])
        print("FN:", counts[(cid, "FN")])

    print("\n저장 위치:", OUTPUT)
    print("오류 목록:", csv_path)


if __name__ == "__main__":
    main()
