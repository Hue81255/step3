
from multiprocessing import freeze_support
from ultralytics import YOLO
import matplotlib.pyplot as plt

from onoff_eval import (
    BEST_PT,
    CLASS_NAMES,
    load_dataset,
    onoff
)

# 강의 자료의 임계값 목록
THRS = [
    0.001, 0.004, 0.01, 0.02,
    0.06, 0.1, 0.2, 0.3
]


def main():
    model = YOLO(str(BEST_PT))

    # 검증셋만 사용
    val_imgs, val_gts = load_dataset("val")

    best = {}
    all_scores = {}

    for cid, name in enumerate(CLASS_NAMES):
        scores = []

        print(f"\n===== {name} =====")

        for t in THRS:
            result = onoff(
                model,
                val_imgs,
                val_gts,
                t,
                cid
            )

            scores.append(result["f1"])

            print(
                f"{name} | thr={t:.3f} | "
                f"F1={result['f1']:.3f} | "
                f"FP={result['fp']} | "
                f"FN={result['fn']}"
            )

        all_scores[name] = scores

        # F1 최고점 선택
        max_f1 = max(scores)

        # 동점이면 낮은 임계값 선택
        best_idx = next(
            i for i, score in enumerate(scores)
            if score == max_f1
        )

        best[name] = THRS[best_idx]

        print(
            f">> {name} 선택 임계값: "
            f"{best[name]} "
            f"(F1={max_f1:.3f})"
        )

    print("\n최종 임계값:", best)

    # F1 곡선 시각화
    for name in CLASS_NAMES:
        plt.plot(
            THRS,
            all_scores[name],
            marker="o",
            label=name
        )

    plt.xlabel("Confidence threshold")
    plt.ylabel("F1-score")
    plt.title("Validation F1 by Threshold")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(
        "val_threshold_f1.png",
        dpi=150
    )

    print("\n그래프 저장 완료: val_threshold_f1.png")

    plt.show()


if __name__ == "__main__":
    freeze_support()
    main()
