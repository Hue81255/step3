
from pathlib import Path
import shutil

DATA_ROOT = Path(
    r"C:\Users\lee\Downloads\사용한 데이터\사용한 데이터"
)

PROJECT_ROOT = Path(r"C:\Users\lee\step3-defect")

TARGET_ID = "URT017_2020_02_08_001_24512"

IMAGE_DIR = DATA_ROOT / "images" / "train"
LABEL_DIR = DATA_ROOT / "labels" / "train"

BACKUP_DIR = PROJECT_ROOT / "leakage_backup"
BACKUP_IMAGES = BACKUP_DIR / "images"
BACKUP_LABELS = BACKUP_DIR / "labels"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main():
    images = sorted(
        p for p in IMAGE_DIR.iterdir()
        if p.is_file()
        and p.suffix.lower() in IMAGE_EXTENSIONS
        and (
            p.stem == TARGET_ID
            or p.stem.startswith(TARGET_ID + "_")
        )
    )

    print("===== 데이터 누수 수정 =====")
    print("이동할 학습 이미지:", len(images))

    if len(images) != 55:
        print("예상 이미지 수(54장)와 다릅니다.")
        print("안전을 위해 작업을 중단합니다.")
        return

    pairs = []

    for image in images:
        label = LABEL_DIR / (image.stem + ".txt")

        if not label.exists():
            print("라벨 누락:", label)
            return

        pairs.append((image, label))

    BACKUP_IMAGES.mkdir(parents=True, exist_ok=True)
    BACKUP_LABELS.mkdir(parents=True, exist_ok=True)

    # 덮어쓰기 방지
    for image, label in pairs:
        if (BACKUP_IMAGES / image.name).exists():
            print("백업 이미지가 이미 존재합니다:", image.name)
            return

        if (BACKUP_LABELS / label.name).exists():
            print("백업 라벨이 이미 존재합니다:", label.name)
            return

    for image, label in pairs:
        shutil.move(str(image), str(BACKUP_IMAGES / image.name))
        shutil.move(str(label), str(BACKUP_LABELS / label.name))

    print("이동 완료!")
    print("이동 이미지:", len(pairs))
    print("이동 라벨:", len(pairs))
    print("백업 위치:", BACKUP_DIR)

    remaining = sum(
        1 for p in IMAGE_DIR.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )

    print("남은 Train 이미지:", remaining)


if __name__ == "__main__":
    main()
