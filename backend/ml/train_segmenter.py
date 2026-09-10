"""U-Net segmenter training — Zenodo Part III, §0.

Split comes from scripts/data_split.py and is never reimplemented, inlined,
hardcoded or temporarily overridden here (§0). Two implementations means one of
them is wrong and nobody knows which.

Trains on 001-120, validates on 121-145. Indices 146-150 are the demo holdout
and reaching for them raises (§9).

Metrics are reported with their split and n: "F1 0.87 (val, n=25)" (§16).
"""

from pathlib import Path


def build_dataset(class_name: str, split: str, data_dir: Path):
    """Resolves indices via data_split.get_split and asserts they are not
    holdout before a single file is opened."""
    ...


def train(data_dir: Path, out_path: Path, epochs: int, batch_size: int, lr: float) -> None: ...


def main() -> None: ...


if __name__ == "__main__":
    main()
