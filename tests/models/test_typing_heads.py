import pandas as pd  # type: ignore[import-untyped]
import pytest

from metaboguard.models.ssl_encoder import SyntheticPrototypeAuthorization
from metaboguard.models.typing_heads import (
    TypingHeadModel,
    TypingHeadNotAuthorizedError,
)


def test_typing_head_rejects_unrecognized_target() -> None:
    with pytest.raises(ValueError, match="unrecognized typing target"):
        TypingHeadModel("cancer_type_v2")


@pytest.mark.parametrize(
    "typing_target",
    ["cancer_type", "diabetes_type_1", "diabetes_type_gestational"],
)
def test_typing_head_fit_always_fails_closed(typing_target: str) -> None:
    with pytest.raises(
        TypingHeadNotAuthorizedError, match="disabled future-development"
    ):
        TypingHeadModel(typing_target).fit(pd.DataFrame(), {})


def test_typing_head_synthetic_prototype_also_fails_closed() -> None:
    with pytest.raises(TypingHeadNotAuthorizedError, match="does not authorize"):
        TypingHeadModel("cancer_type").fit_synthetic_prototype(
            pd.DataFrame(), SyntheticPrototypeAuthorization("approval-2026")
        )
