"""Synthetic-truth attribution benchmark.

Generates releases with a known culprit, drifts them forward, then runs the
backward pipeline and asks whether the true vessel is ranked first and whether
the verdict band is right. This is the only place attribution accuracy can be
measured, because no real spill in the dataset carries a labelled culprit.

Reports top-1 accuracy and the UNATTRIBUTED rate against the false-accusation
rate — the second number matters more than the first (§2.4).
"""

from pathlib import Path


def generate_case(seed: int) -> dict: ...


def run_benchmark(n_cases: int, seed: int, out_path: Path) -> dict[str, float]: ...


def main() -> None: ...


if __name__ == "__main__":
    main()
