
from pathlib import Path
from multiprocessing import freeze_support
import shutil
import yaml

from ultralytics import YOLO


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

LEAK_ID = "URT017_2020_02_08_001_24512"

# 39장 평가용 임시 데이터셋
CLEAN_ROOT = BASE_DIR / "test_clean_39"

IMAGE_DIR = DATA_ROOT / "images" / "test"
LABEL_DIR = DATA_ROOT / "labels" / "test"

CLEAN_IMAGE_DIR = CLEAN_ROOT / "images" / "test"
CLEAN_LABEL_DIR = CLEAN_ROOT / "labels" / "test"


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"모델 없음: {MODEL_PATH}")

    images = sorted(
        p for p in IMAGE_DIR.iterdir()
        if p.is_file()
        and p.suffix.lower() in {
            ".jpg", ".jpeg", ".png"
        }
    )

    excluded = [
        p for p in images
        if p.stem == LEAK_ID
        or p.stem.startswith(LEAK_ID + "_")
    ]

    print("===== 39장 Mask mAP 평가 준비 =====")
    print("전체 테스트 이미지:", len(images))
    print("제외할 이미지:", len(excluded))

    for p in excluded:
        print(" -", p.name)

    if len(images) != 40 or len(excluded) != 1:
        raise RuntimeError(
            "테스트 이미지 40장 중 중복 이미지 "
            "1장이 확인되지 않아 중단합니다."
        )

    selected = [
        p for p in images
        if p not in excluded
    ]

    # 복사 전 모든 라벨 확인
    for image in selected:
        label = LABEL_DIR / f"{image.stem}.txt"

        if not label.exists():
            raise FileNotFoundError(
                f"라벨 파일 없음: {label}"
            )

    # 기존 임시 폴더가 있다면 중단
    # 기존 파일을 실수로 삭제하지 않도록 보호
    if CLEAN_ROOT.exists():
        raise RuntimeError(
            f"폴더가 이미 존재합니다: {CLEAN_ROOT}\n"
            "기존 내용을 확인한 뒤 다시 실행하세요."
        )

    CLEAN_IMAGE_DIR.mkdir(parents=True)
    CLEAN_LABEL_DIR.mkdir(parents=True)

    for image in selected:
        label = LABEL_DIR / f"{image.stem}.txt"

        shutil.copy2(
            image,
            CLEAN_IMAGE_DIR / image.name
        )

        shutil.copy2(
            label,
            CLEAN_LABEL_DIR / label.name
        )

    # Ultralytics가 인식할 수 있는 데이터셋 설정
    clean_yaml = CLEAN_ROOT / "data.yaml"

    config = {
        "path": str(CLEAN_ROOT),
        "train": "images/test",
        "val": "images/test",
        "test": "images/test",
        "names": {
            0: "crack",
            1: "efflorescence",
            2: "spall_detach",
        },
    }

    with clean_yaml.open(
        "w", encoding="utf-8"
    ) as f:
        yaml.safe_dump(
            config,
            f,
            allow_unicode=True,
            sort_keys=False
        )

    print("평가용 이미지:", len(selected))
    print("평가용 데이터셋:", CLEAN_ROOT)

    # 기존 모델 사용: 학습하지 않음
    model = YOLO(str(MODEL_PATH))

    metrics = model.val(
        data=str(clean_yaml),
        split="test",
        imgsz=960,
        device=0,
        workers=0,
        plots=True,
        project=str(BASE_DIR / "runs" / "segment"),
        name="eval-clean-39",
        exist_ok=False
    )

    print("\n===== 최종 39장 평가 결과 =====")
    print(f"Box mAP50:       {metrics.box.map50:.4f}")
    print(f"Mask mAP50:      {metrics.seg.map50:.4f}")
    print(f"Mask mAP50-95:   {metrics.seg.map:.4f}")
    print("평가 이미지 수:", len(selected))


if __name__ == "__main__":
    freeze_support()
    main()
