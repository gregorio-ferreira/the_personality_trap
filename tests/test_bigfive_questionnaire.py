from typing import Any, Dict, Optional, Tuple

import pandas as pd

from personas_backend.questionnaire.bigfive import BigFive


class BigFiveNoDB(BigFive):
    """Test double that skips database writes."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.saved_df: Optional[pd.DataFrame] = None

    def _insert_db_eval_questionnaires(  # type: ignore[override]
        self, exp_df: pd.DataFrame, experiment_id: int
    ) -> pd.DataFrame:
        self.saved_df = exp_df.copy()
        return exp_df


class DummyWrapper:
    def generate_conversation(
        self, **kwargs: Any
    ) -> Tuple[Dict[str, Any], Any, Dict[str, Any]]:  # pragma: no cover
        raise NotImplementedError


class DummyExpHandler:
    def record_request_metadata_and_status(
        self, *args: Any, **kwargs: Any
    ) -> None:  # pragma: no cover
        pass


def build_bigfive_df() -> pd.DataFrame:
    """Build a minimal valid BigFive answer set with variance."""
    answers = {"question_number": list(range(1, 45)), "answer": []}
    pattern = [1, 2, 3, 4, 5]
    for i in range(44):
        answers["answer"].append(pattern[i % 5])
    return pd.DataFrame(answers)


def build_bigfive_json() -> Dict[str, Any]:
    pattern = [1, 2, 3, 4, 5]
    payload: Dict[str, Any] = {
        str(i + 1): pattern[i % len(pattern)] for i in range(BigFive.NUM_QUESTIONS)
    }
    payload["explanation"] = "Validated response"
    return payload


def test_bigfive_validate_and_score() -> None:  # type: ignore
    bf = BigFive(
        DummyWrapper(),
        DummyExpHandler(),
        db_conn=None,
        logger=None,
        schema="personality_trap",
    )
    df = build_bigfive_df()
    assert bf._validate_answers(df.copy()) is True
    low_var = df.copy()
    low_var["answer"] = 3
    assert bf._validate_answers(low_var) is False
    scores = bf.calculate_scores(experiment_id=123, exp_df=df)
    assert scores is not None
    expected_traits = {
        "extraversion",
        "agreeableness",
        "conscientiousness",
        "neuroticism",
        "openness",
    }
    assert expected_traits.issubset(scores.keys())
    for v in scores.values():
        if v is not None:
            assert 1 <= v <= 5


def test_bigfive_parse_valid_response_roundtrip() -> None:
    bf = BigFiveNoDB(
        DummyWrapper(),
        DummyExpHandler(),
        db_conn=None,
        logger=None,
        schema="personality_trap",
    )
    json_payload = build_bigfive_json()
    explanation = bf.parse_valid_response(
        42,
        json_payload,
        auto_record=False,
    )
    assert explanation == json_payload["explanation"]
    assert bf.saved_df is not None
    assert len(bf.saved_df) == BigFive.NUM_QUESTIONS
