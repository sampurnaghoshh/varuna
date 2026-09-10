"""LightGBM oil vs look-alike training — Zenodo Part III, §0.

Split comes from scripts/data_split.py, same enforcement as the segmenter: train
001-120, validate 121-145, holdout 146-150 raises (§0, §9).

Hyperparameters are selected on val, which makes val an optimistic estimate.
Say so when reporting (§16).
"""

from pathlib import Path


def build_feature_table(class_name: str, split: str, data_dir: Path):
    """Resolves indices via data_split.get_split and asserts not-holdout."""
    ...


def train(data_dir: Path, out_path: Path) -> None: ...


def main() -> None: ...


if __name__ == "__main__":
    main()
