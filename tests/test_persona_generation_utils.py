import pandas as pd  # type: ignore
from personas_backend.personas_generation.personas_generation import (
    PersonaGenerator,
)


def test_compute_borderline_inverse_and_direct():  # type: ignore
    # Build minimal baseline questionnaire
    baseline = pd.DataFrame(
        {
            "personality_id": [1, 1, 1, 1],
            "category": ["E", "N", "P", "L"],
            "question_number": [1, 2, 3, 4],
            "question": ["q1", "q2", "q3", "q4"],
            "key": [True, False, True, False],
            "answer": [True, True, False, False],
        }
    )
    df_inv = PersonaGenerator.compute_borderline(baseline, "E", True)
    assert not df_inv.loc[df_inv["category"] == "E", "answer"].iloc[0]
    df_dir = PersonaGenerator.compute_borderline(baseline, "N", False)
    assert not df_dir.loc[df_dir["category"] == "N", "answer"].iloc[0]


def test_compute_borderline_multiple_rows_category():  # type: ignore
    baseline = pd.DataFrame(
        {
            "personality_id": [1, 1, 1, 1, 1],
            "category": ["E", "E", "N", "P", "L"],
            "question_number": [1, 2, 3, 4, 5],
            "question": ["q1", "q2", "q3", "q4", "q5"],
            "key": [True, False, True, False, True],
            "answer": [False, False, False, False, False],
        }
    )
    df_inv = PersonaGenerator.compute_borderline(baseline, "E", True)
    e_answers = df_inv.loc[df_inv["category"] == "E", "answer"].tolist()
    # Inversion relative to key (True->False, False->True)
    assert e_answers == [False, True]


def test_compute_borderline_absent_category_no_change():  # type: ignore
    baseline = pd.DataFrame(
        {
            "personality_id": [1, 1],
            "category": ["E", "N"],
            "question_number": [1, 2],
            "question": ["q1", "q2"],
            "key": [True, False],
            "answer": [True, True],
        }
    )
    df_out = PersonaGenerator.compute_borderline(baseline, "P", True)
    # No category 'P' so identical answers
    assert df_out["answer"].tolist() == baseline["answer"].tolist()


def test_compute_borderline_idempotent_given_same_keys():  # type: ignore
    baseline = pd.DataFrame(
        {
            "personality_id": [1, 1, 1],
            "category": ["E", "N", "P"],
            "question_number": [1, 2, 3],
            "question": ["q1", "q2", "q3"],
            "key": [True, False, True],
            "answer": [True, True, False],
        }
    )
    first = PersonaGenerator.compute_borderline(baseline, "E", True)
    second = PersonaGenerator.compute_borderline(first, "E", True)
    # Because logic derives from 'key', repeated application stable
    first_e = first.loc[first["category"] == "E", "answer"].tolist()
    second_e = second.loc[second["category"] == "E", "answer"].tolist()
    assert first_e == second_e


def test_compute_borderline_bool_dtype_and_numeric_keys():  # type: ignore
    baseline = pd.DataFrame(
        {
            "personality_id": [1, 1, 1, 1],
            "category": ["E", "N", "P", "L"],
            "question_number": [1, 2, 3, 4],
            "question": ["q1", "q2", "q3", "q4"],
            "key": [1, 0, 1, 0],  # ints
            "answer": [0, 0, 0, 0],
        }
    )
    out = PersonaGenerator.compute_borderline(baseline, "N", False)
    # All answers bool dtype
    assert out["answer"].dtype == bool
    # Category N takes key (0 -> False)
    assert not out.loc[out["category"] == "N", "answer"].iloc[0]
