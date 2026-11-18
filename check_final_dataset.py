
from pathlib import Path
from collections import defaultdict, Counter
import re

DATA_ROOT = Path(
    r"C:\Users\lee\Downloads\사용한 데이터\사용한 데이터"
)

SPLITS = ["train", "val", "test"]
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# 알려진 증강 파일명 패턴
AUG_PATTERN = re.compile(
    r"(?i)(?:^|[_-])(?:aug|augment|augmented|"
    r"flip|flipped|rotate|rotated|rotation|"
    r"mosaic|mixup|brightness|contrast|"
    r"blur|noise|crop|cropped|scale|scaled)"
    r"(?:[_-]|\d|$)"
)


def get_original_id(filename):
    stem = Path(filename).stem

    # Roboflow 파일명에서 .rf.해시 제거
    stem = re.sub(
        r"(?i)_jpg\.rf\.[a-f0-9]+$",
        "",
        stem
    )

    # 일반적인 증강 접미사 제거
    stem = re.split(
        r"(?i)(?:_aug(?:mented)?|_flip|_rotate|"
        r"_rotated|_brightness|_contrast|"
        r"_blur|_noise|_crop|_scale)",
        stem
    )[0]

    return stem


def main():
    original_splits = defaultdict(set)
    original_files = defaultdict(list)
    split_counts = {}

    print("===== 데이터셋 검사 시작 =====")

    for split in SPLITS:
        image_dir = DATA_ROOT / "images" / split

        if not image_dir.exists():
            print(f"[오류] 폴더 없음: {image_dir}")
            return

        files = sorted(
            p for p in image_dir.iterdir()
            if p.is_file()
            and p.suffix.lower() in EXTENSIONS
        )

        split_counts[split] = len(files)

        augmented = []

        for path in files:
            original_id = get_original_id(path.name)

            original_splits[original_id].add(split)
            original_files[original_id].append(
                (split, path.name)
            )

            if AUG_PATTERN.search(path.stem):
                augmented.append(path.name)

        print(f"\n[{split}]")
        print("이미지 수:", len(files))
        print("증강 의심 파일:", len(augmented))

        for name in augmented[:10]:
            print("  -", name)

    # 서로 다른 split에 동일 원본 ID가 있는지 확인
    overlaps = {
        original_id: splits
        for original_id, splits in original_splits.items()
        if len(splits) > 1
    }

    print("\n===== 원본 단위 분할 검사 =====")
    print("원본 ID 중복:", len(overlaps))

    for original_id, splits in list(overlaps.items())[:10]:
        print(
            f"\n원본 ID: {original_id}"
        )
        print("포함 split:", sorted(splits))

        for split, filename in original_files[original_id]:
            print(" ", split, filename)

    # 동일 split 내부의 파생 이미지 개수
    counts = Counter(
        tuple(sorted(splits))
        for splits in original_splits.values()
    )

    print("\n===== 요약 =====")
    print("Train:", split_counts["train"])
    print("Val:", split_counts["val"])
    print("Test:", split_counts["test"])
    print("서로 다른 split 간 원본 ID 중복:", len(overlaps))

    if len(overlaps) == 0:
        print("원본 ID 기준 분할 검사 통과")
    else:
        print("주의: 원본 ID가 여러 split에 존재함")

    print(
        "\n주의: 파일명 기반 검사이므로 "
        "이름이 다른 동일·유사 이미지까지 "
        "완전히 검출하지는 못합니다."
    )


if __name__ == "__main__":
    main()
