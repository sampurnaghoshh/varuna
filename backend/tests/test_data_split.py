"""The holdout guard is a safety property (§9). These tests are the proof."""

import pytest

from scripts.data_split import (
    CLASS_NAMES,
    HOLDOUT_RANGE,
    HoldoutViolationError,
    UnknownClassError,
    assert_not_holdout,
    get_split,
    is_holdout,
    split_of,
    training_indices,
)


@pytest.mark.parametrize("class_name", CLASS_NAMES)
def test_split_ranges_are_exact(class_name: str) -> None:
    split = get_split(class_name)
    assert split["train"] == list(range(1, 121))
    assert split["val"] == list(range(121, 146))
    assert split["holdout"] == list(range(146, 151))


@pytest.mark.parametrize("class_name", CLASS_NAMES)
def test_splits_are_disjoint_and_cover_150(class_name: str) -> None:
    split = get_split(class_name)
    combined = split["train"] + split["val"] + split["holdout"]
    assert len(combined) == 150
    assert len(set(combined)) == 150


def test_unknown_class_raises() -> None:
    with pytest.raises(UnknownClassError):
        get_split("oil_free")


@pytest.mark.parametrize(
    ("index", "expected"),
    [(1, "train"), (120, "train"), (121, "val"), (145, "val"), (146, "holdout"), (150, "holdout")],
)
def test_split_boundaries(index: int, expected: str) -> None:
    assert split_of(index) == expected


@pytest.mark.parametrize("index", range(HOLDOUT_RANGE[0], HOLDOUT_RANGE[1] + 1))
def test_every_holdout_index_raises(index: int) -> None:
    assert is_holdout(index)
    with pytest.raises(HoldoutViolationError):
        assert_not_holdout([index], context="test")


def test_guard_passes_train_and_val() -> None:
    assert_not_holdout(range(1, 146), context="test")


def test_guard_names_the_offenders() -> None:
    with pytest.raises(HoldoutViolationError) as excinfo:
        assert_not_holdout([10, 147, 150], context="train_segmenter")
    message = str(excinfo.value)
    assert "147" in message
    assert "150" in message
    assert "train_segmenter" in message


def test_training_indices_refuses_holdout_split() -> None:
    with pytest.raises(HoldoutViolationError):
        training_indices("oil", "holdout", context="test")


@pytest.mark.parametrize("split", ["train", "val"])
def test_training_indices_allows_train_and_val(split: str) -> None:
    assert training_indices("oil", split, context="test")
