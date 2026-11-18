
from pathlib import Path
from ultralytics import YOLO

# ===== 경로 설정 =====
BASE_DIR = Path(r"C:\Users\lee\step3-defect")
DATA_ROOT = Path(
    r"C:\Users\lee\Downloads\사용한 데이터\사용한 데이터"
)

MODEL_PATH = (
    BASE_DIR
    / "runs"
    / "segment"
    / "runs"
    / "step3-5"
    / "weights"
    / "best.pt"
)

IMAGE_DIR = DATA_ROOT / "images" / "test"
LABEL_DIR = DATA_ROOT / "labels" / "test"

# 학습 데이터와 원본이 중복된 테스트 이미지
LEAK_ID = "URT017_2020_02_08_001_24512"

# 검증셋에서 결정한 임계값 (변경하지 않음)
THRESHOLDS = {
    0: 0.2,   # crack
    1: 0.02,  # efflorescence
    2: 0.2,   # spall_detach
}

CLASS_NAMES = {
    0: "crack",
    1: "efflorescence",
    2: "spall_detach",
}


def calculate_metrics(tp, fp, fn):
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall else 0.0
    )

    return precision, recall, f1


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"모델 없음: {MODEL_PATH}")

    model = YOLO(str(MODEL_PATH))

    images = sorted(
        p for p in IMAGE_DIR.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )

    excluded = [
        p for p in images
        if p.stem == LEAK_ID
        or p.stem.startswith(LEAK_ID + "_")
    ]

    test_images = [
        p for p in images
        if p not in excluded
    ]

    print("===== 테스트셋 중복 제거 평가 =====")
    print("전체 테스트 이미지:", len(images))
    print("제외 이미지:", len(excluded))
    for p in excluded:
        print(" -", p.name)
    print("실제 평가 이미지:", len(test_images))

    # 잘못된 이미지를 제외하는 상황 방지
    if len(images) != 40 or len(excluded) != 1:
        raise RuntimeError(
            "테스트 이미지 40장 중 중복 1장이 확인되지 않아 중단합니다."
        )

    counts = {
        cid: {"TP": 0, "FP": 0, "FN": 0, "TN": 0}
        for cid in CLASS_NAMES
    }

    for index, image_path in enumerate(test_images, 1):
        label_path = LABEL_DIR / (image_path.stem + ".txt")

        if not label_path.exists():
            raise FileNotFoundError(f"라벨 없음: {label_path}")

        # 이미지별 정답 클래스
        gt_classes = set()

        for line in label_path.read_text(
            encoding="utf-8"
        ).splitlines():
            parts = line.strip().split()
            if parts:
                gt_classes.add(int(parts[0]))

        # 기존 평가와 동일한 추론 설정
        result = model.predict(
            source=str(image_path),
            conf=0.001,
            imgsz=960,
            device=0,
            verbose=False
        )[0]

        pred_classes = set()

        if result.boxes is not None:
            for cls, conf in zip(
                result.boxes.cls.tolist(),
                result.boxes.conf.tolist()
            ):
                cid = int(cls)

                if (
                    cid in THRESHOLDS
                    and conf >= THRESHOLDS[cid]
                ):
                    pred_classes.add(cid)

        for cid in CLASS_NAMES:
            gt_on = cid in gt_classes
            pred_on = cid in pred_classes

            if gt_on and pred_on:
                counts[cid]["TP"] += 1
            elif not gt_on and pred_on:
                counts[cid]["FP"] += 1
            elif gt_on and not pred_on:
                counts[cid]["FN"] += 1
            else:
                counts[cid]["TN"] += 1

        print(f"[{index}/{len(test_images)}] {image_path.name}")

    print("\n===== 최종 ON/OFF 평가 =====")

    f1_scores = []

    for cid, name in CLASS_NAMES.items():
        c = counts[cid]

        precision, recall, f1 = calculate_metrics(
            c["TP"], c["FP"], c["FN"]
        )

        f1_scores.append(f1)

        print(f"\n{name}")
        print(f"Threshold: {THRESHOLDS[cid]}")
        print(
            f"TP={c['TP']} FP={c['FP']} "
            f"FN={c['FN']} TN={c['TN']}"
        )
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1:        {f1:.4f}")

    macro_f1 = sum(f1_scores) / len(f1_scores)

    print("\n==============================")
    print(f"Test images: {len(test_images)}")
    print(f"Macro F1: {macro_f1:.4f}")
    print("==============================")


if __name__ == "__main__":
    main()
