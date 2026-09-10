"""Detection evaluation — val split only (121-145, n = 25 per class).

Never evaluates on holdout. 146-150 is n = 5 per class: a metric computed on it
is noise with a decimal point, and holdout exists for demo-scene provenance
alone (§16).

Every figure this prints carries its split name and n.
"""

from pathlib import Path


def evaluate(weights_path: Path, data_dir: Path) -> dict[str, float]: ...


def main() -> None: ...


if __name__ == "__main__":
    main()
