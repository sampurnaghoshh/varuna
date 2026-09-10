"""THE data split. Zenodo record 13761290, Part III. See §0 of CLAUDE.md.

Part III holds 150 images per class (oil, look-alike, oil-free). The split is a
function of the filename index and is identical for every class:

    001-120   train
    121-145   val
    146-150   HOLDOUT - demo scenes only (SC-01 / SC-02 / SC-03, §12)

This module is the single place those ranges exist. ml/train_segmenter.py and
ml/train_discriminator.py import it; they never reimplement, inline, hardcode or
temporarily override it. Two implementations means one of them is wrong and
nobody knows which.

Enforcement raises. A training or validation path that reaches for 146-150 fails
loudly and stops - not a warning, not a silent filter. A model that has seen the
demo scenes makes the demo a lie (§9).

data/scenes/** stays split-unaware by design: the demo pipeline is *supposed* to
load holdout scenes, so there is no guard at scene ingest. The split is a
training-data concern and lives at the training-data loader.
"""

from collections.abc import Iterable
from typing import Final, Literal

CLASS_NAMES: Final[tuple[str, ...]] = ("oil", "look-alike", "oil-free")
IMAGES_PER_CLASS: Final[int] = 150

TRAIN_RANGE: Final[tuple[int, int]] = (1, 120)
VAL_RANGE: Final[tuple[int, int]] = (121, 145)
HOLDOUT_RANGE: Final[tuple[int, int]] = (146, 150)

SplitName = Literal["train", "val", "holdout"]


class HoldoutViolationError(RuntimeError):
    """Raised when a training or validation path reaches for a holdout index."""


class UnknownClassError(ValueError):
    """Raised for a class name that is not one of the three Part III folders."""


def _validate_class(class_name: str) -> str:
    if class_name not in CLASS_NAMES:
        raise UnknownClassError(
            f"Unknown class {class_name!r}. Part III has exactly {CLASS_NAMES}."
        )
    return class_name


def get_split(class_name: str) -> dict[SplitName, list[int]]:
    """Return the train / val / holdout index lists for one Part III class."""
    _validate_class(class_name)
    return {
        "train": list(range(TRAIN_RANGE[0], TRAIN_RANGE[1] + 1)),
        "val": list(range(VAL_RANGE[0], VAL_RANGE[1] + 1)),
        "holdout": list(range(HOLDOUT_RANGE[0], HOLDOUT_RANGE[1] + 1)),
    }


def split_of(index: int) -> SplitName:
    """Which split an index belongs to. Raises for an index outside 1-150."""
    if not 1 <= index <= IMAGES_PER_CLASS:
        raise ValueError(f"Index {index} is outside Part III range 1-{IMAGES_PER_CLASS}.")
    if index <= TRAIN_RANGE[1]:
        return "train"
    if index <= VAL_RANGE[1]:
        return "val"
    return "holdout"


def is_holdout(index: int) -> bool:
    return HOLDOUT_RANGE[0] <= index <= HOLDOUT_RANGE[1]


def assert_not_holdout(indices: Iterable[int], context: str) -> None:
    """Guard every training and validation data path with this.

    Raises HoldoutViolationError listing the offending indices. There is no
    override argument and there will not be one: the point of the guard is that
    it cannot be relaxed when a run is inconvenient (§9).
    """
    offenders = sorted({int(i) for i in indices if is_holdout(int(i))})
    if offenders:
        raise HoldoutViolationError(
            f"{context}: indices {offenders} are in the demo holdout range "
            f"{HOLDOUT_RANGE[0]}-{HOLDOUT_RANGE[1]} and must never be trained or "
            f"validated on (CLAUDE.md §0, §9)."
        )


def training_indices(class_name: str, split: SplitName, context: str) -> list[int]:
    """Indices for a training-time split, guarded. Asking for 'holdout' here
    raises: no training-data loader has a legitimate reason to want it."""
    indices = get_split(class_name)[split]
    assert_not_holdout(indices, context)
    return indices
