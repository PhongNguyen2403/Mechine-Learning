import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parents[1]))

from app.modeling import ARTIFACT_DIR, DATASET_PATH, load_dataset


def main() -> None:
    features, labels, feature_names = load_dataset(DATASET_PATH)
    report = {
        "dataset": DATASET_PATH.name,
        "rows": int(features.shape[0]),
        "feature_count": int(features.shape[1]),
        "feature_names": feature_names,
        "missing_values": 0,
        "class_distribution": {
            str(label): int(np.sum(labels == label)) for label in sorted(np.unique(labels))
        },
        "summary": {
            name: {
                "min": round(float(features[:, index].min()), 4),
                "max": round(float(features[:, index].max()), 4),
                "mean": round(float(features[:, index].mean()), 4),
                "std": round(float(features[:, index].std()), 4),
            }
            for index, name in enumerate(feature_names)
        },
    }
    ARTIFACT_DIR.mkdir(exist_ok=True)
    report_path = ARTIFACT_DIR / "eda_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved EDA report to {report_path}")


if __name__ == "__main__":
    main()