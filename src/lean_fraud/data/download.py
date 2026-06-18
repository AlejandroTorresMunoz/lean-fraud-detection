"""Download the IEEE-CIS Fraud Detection training data into data/raw/ via the Kaggle API.

Requires a free Kaggle account + API token at ~/.kaggle/kaggle.json (or KAGGLE_USERNAME /
KAGGLE_KEY env vars) AND a one-time acceptance of the competition rules at
https://www.kaggle.com/c/ieee-fraud-detection/rules — still possible even though the 2019
competition is closed. We only fetch the *labelled* train files; the competition test set has no
public labels, so build_sequences makes its own time-based split on the train data.

Usage: python -m lean_fraud.data.download
"""

from __future__ import annotations

import zipfile
from pathlib import Path

from lean_fraud.config import load_config

COMPETITION = "ieee-fraud-detection"
FILES = ["train_transaction.csv", "train_identity.csv"]


def main() -> None:
    cfg = load_config()
    raw_dir = Path(cfg["dataset"]["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as exc:  # pragma: no cover - depends on the optional `data` group
        raise SystemExit("kaggle not installed — run `uv sync --group data`.") from exc

    api = KaggleApi()
    api.authenticate()  # reads ~/.kaggle/kaggle.json or KAGGLE_USERNAME / KAGGLE_KEY

    for fname in FILES:
        if (raw_dir / fname).exists():
            print(f"[download] {fname} already present, skipping")
            continue
        print(f"[download] fetching {fname} ...")
        try:
            api.competition_download_file(COMPETITION, fname, path=str(raw_dir), quiet=False)
        except Exception as exc:  # noqa: BLE001 - surface a clear, actionable hint
            raise SystemExit(
                f"Could not download {fname} ({exc}). Check that your Kaggle token is set and that "
                f"you accepted the rules at https://www.kaggle.com/c/{COMPETITION}/rules"
            ) from exc

        zipped = raw_dir / f"{fname}.zip"
        if zipped.exists():
            with zipfile.ZipFile(zipped) as zf:
                zf.extractall(raw_dir)
            zipped.unlink()

    print(f"[download] done -> {raw_dir.resolve()}")


if __name__ == "__main__":
    main()
