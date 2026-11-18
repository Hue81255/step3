
from pathlib import Path
from multiprocessing import freeze_support
from ultralytics import YOLO

BASE_DIR = Path(r"C:\Users\lee\step3-defect")
DATA_ROOT = Path(
    r"C:\Users\lee\Downloads\사용한 데이터\사용한 데이터"
)

BEST_PT = (
    BASE_DIR / "runs" / "segment" / "runs"
    / "step3-5" / "weights" / "best.pt"
)

CLASS_NAMES = [
    "crack",
    "efflorescence",
    "spall_detach"
]


def load_dataset(split):
    img_dir = DATA_ROOT / "images" / split
    lbl_dir = DATA_ROOT / "labels" / split

    paths = sorted(img_dir.glob("*.jpg"))
    gts = []

    for p in paths:
        label_path = lbl_dir / (p.stem + ".txt")
        classes = set()

        for line in label_path.read_text(
            encoding="utf-8-sig"
        ).splitlines():
            if line.strip():
                classes.add(int(line.split()[0]))

        gts.append(classes)

    return paths, gts


def onoff(model, paths, gts, thr, cid_target):
    tp = fp = fn = tn = 0

    for p, gt in zip(paths, gts):
        r = model.predict(
            str(p),
            imgsz=960,
            conf=0.001,
            device=0,
            verbose=False
        )[0]

        conf = r.boxes.conf.cpu().numpy()
        cid = r.boxes.cls.cpu().numpy().astype(int)

        pred_on = bool(
            ((cid == cid_target) & (conf >= thr)).any()
        )

        gt_on = cid_target in gt

        if gt_on and pred_on:
            tp += 1
        elif not gt_on and pred_on:
            fp += 1
        elif gt_on and not pred_on:
            fn += 1
        else:
            tn += 1

    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0
    )

    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


def main():
    model = YOLO(str(BEST_PT))

    # STEP 2에서는 검증셋만 사용
    val_imgs, val_gts = load_dataset("val")

    # 강의 자료의 초기 임계값 0.5
    for cid, name in enumerate(CLASS_NAMES):
        result = onoff(
            model, val_imgs, val_gts, 0.5, cid
        )

        print(f"\n[{name}]")
        print("TP:", result["tp"])
        print("FP:", result["fp"])
        print("FN:", result["fn"])
        print("TN:", result["tn"])
        print("Precision:", round(result["precision"], 3))
        print("Recall:", round(result["recall"], 3))
        print("F1:", round(result["f1"], 3))


if __name__ == "__main__":
    freeze_support()
    main()
