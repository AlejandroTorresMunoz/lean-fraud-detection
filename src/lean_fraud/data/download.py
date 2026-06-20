"""Download the Sparkov synthetic credit-card fraud dataset into data/raw/ via the Kaggle API.

This is a public Kaggle *dataset* (not a competition), so it downloads with just an API token —
no competition rules / phone verification needed. It ships two CSVs (fraudTrain.csv, fraudTest.csv)
with one row per card transaction: cc_num (card -> pseudo-user), unix_time, amt, category, merchant
and an is_fraud label — exactly what build_sequences needs to build per-user transaction sequences.
We merge both files and make our own strict time-based split downstream.

Requires a Kaggle token: KAGGLE_API_TOKEN in .env (loaded below) or ~/.kaggle/kaggle.json.

Usage: python -m lean_fraud.data.download
"""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

from lean_fraud.config import load_config

DATASET = "kartik2112/fraud-detection"
FILES = ["fraudTrain.csv", "fraudTest.csv"]


def main() -> None:
    # Pull KAGGLE_API_TOKEN (and friends) from .env into the process environment so the Kaggle
    # client can authenticate without exporting secrets into the shell.
    load_dotenv()

    cfg = load_config()
    raw_dir = Path(cfg["dataset"]["raw_dir"])
    raw_dir.mkdir(parents=True, exist_ok=True)

    if all((raw_dir / fname).exists() for fname in FILES):
        print(f"[download] {', '.join(FILES)} already present, skipping")
        return

    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as exc:  # pragma: no cover - depends on the optional `data` group
        raise SystemExit("kaggle not installed — run `uv sync --group data`.") from exc

    api = KaggleApi()
    # Picks up KAGGLE_API_TOKEN (new KGAT_ token), else KAGGLE_USERNAME / KAGGLE_KEY,
    # else ~/.kaggle/kaggle.json.
    api.authenticate()

    print(f"[download] fetching {DATASET} ...")
    try:
        api.dataset_download_files(DATASET, path=str(raw_dir), unzip=True, quiet=False)
    except Exception as exc:  # noqa: BLE001 - surface a clear, actionable hint
        raise SystemExit(
            f"Could not download {DATASET} ({exc}). Check that your Kaggle token is set "
            "(KAGGLE_API_TOKEN in .env)."
        ) from exc

    print(f"[download] done -> {raw_dir.resolve()}")


if __name__ == "__main__":
    main()
