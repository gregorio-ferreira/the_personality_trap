import pandas as pd
import pytest

from personas_backend.core.enums import ModelID, PersonaGenerationCondition
from personas_backend.personas_generation.personas_generation import (
    PersonaGenerator,
)


class DummyDB:
    pass


class DummyPopulation:
    def check_existing_personality(self, *args, **kwargs):
        return False

    def save_generated_persona(self, *args, **kwargs):
        pass


def _minimal_frames():
    population_df = pd.DataFrame({"persona_id": [1], "description": ["d"]})
    exp_df = pd.DataFrame(
        {
            "personality_id": [1],
            "category": ["N"],
            "question_number": [1],
            "question": ["q1"],
            "key": [True],
            "answer": [False],
            "experiment_id": [1],
        }
    )
    return population_df, exp_df


def test_generate_personas_rejects_experimental_model():
    population_df, exp_df = _minimal_frames()
    gen = PersonaGenerator(DummyDB(), DummyPopulation(), expected_schema={})
    with pytest.raises(ValueError):
        gen.generate_personas(
            models=[ModelID.GPT4O_OLD],
            population_df=population_df,
            new_population_table="t",
            exp_df=exp_df,
            allow_experimental=False,
        )


def test_generate_borderline_personas_rejects_experimental_condition():
    population_df, exp_df = _minimal_frames()
    gen = PersonaGenerator(DummyDB(), DummyPopulation(), expected_schema={})
    with pytest.raises(ValueError):
        gen.generate_borderline_personas(
            model=[ModelID.GPT4O],
            borderline_list=[PersonaGenerationCondition.MIN_L],
            personas_list=[1],
            population_df=population_df,
            new_population_table="t",
            exp_df=exp_df,
            allow_experimental=False,
        )


def test_generate_borderline_personas_smoke_max_n(monkeypatch):
    population_df, exp_df = _minimal_frames()
    gen = PersonaGenerator(DummyDB(), DummyPopulation(), expected_schema={})

    def fake_process(*args, **kwargs):
        return pd.DataFrame({"ref_personality_id": [1], "repetitions": [1]})

    monkeypatch.setattr(gen, "_process_persona", fake_process)
    gen.generate_borderline_personas(
        model=[ModelID.GPT4O],
        borderline_list=[PersonaGenerationCondition.MAX_N],
        personas_list=[1],
        population_df=population_df,
        new_population_table="t",
        exp_df=exp_df,
        allow_experimental=False,
        repetitions=1,
        max_workers=1,
    )
