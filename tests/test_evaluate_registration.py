from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pytest
from personas_backend.db.models import ExperimentsGroup, ExperimentsList, Persona
from personas_backend.evaluate_questionnaire import registration
from sqlmodel import select


@dataclass
class _RecordingGroupCall:
    description: str
    system_role: str
    base_prompt: str
    translated: bool
    temperature: float
    top_p: float
    ii_repeated: bool


class RecordingExperimentGroupHandler:
    """Test double that persists experiment groups into the in-memory database."""

    def __init__(self, session_factory):
        self.session_factory = session_factory
        self.calls: List[_RecordingGroupCall] = []
        self._next_id = 1

    def register_new_experiments_group(
        self,
        *,
        description: str,
        system_role: str,
        base_prompt: str,
        translated: bool,
        temperature: float,
        top_p: float,
        ii_repeated: int,
        schema: Optional[str] = None,
    ) -> int:
        group_id = self._next_id
        self._next_id += 1
        with self.session_factory() as session:
            group = ExperimentsGroup(
                experiments_group_id=group_id,
                description=description,
                system_role=system_role,
                base_prompt=base_prompt,
                translated=translated,
                temperature=temperature,
                top_p=top_p,
                ii_repeated=bool(ii_repeated),
            )
            session.add(group)
            session.commit()
        self.calls.append(
            _RecordingGroupCall(
                description=description,
                system_role=system_role,
                base_prompt=base_prompt,
                translated=translated,
                temperature=temperature,
                top_p=top_p,
                ii_repeated=bool(ii_repeated),
            )
        )
        return group_id


class RecordingExperimentHandler:
    """Test double that inserts experiments while tracking payloads."""

    def __init__(self, session_factory):
        self.session_factory = session_factory
        self.calls: List[Dict[str, Any]] = []
        self._next_id = 1

    def register_experiment(
        self, my_experiment: Dict[str, Any], schema: Optional[str] = None
    ) -> int:
        experiment_id = self._next_id
        self._next_id += 1
        payload = dict(my_experiment)
        with self.session_factory() as session:
            experiment = ExperimentsList(
                experiment_id=experiment_id,
                **{k: v for k, v in payload.items() if v is not None},
            )
            session.add(experiment)
            session.commit()
        payload["experiment_id"] = experiment_id
        self.calls.append(payload)
        return experiment_id


def test_fetch_personas_filters_by_model_and_population(session_factory):
    with session_factory() as session:
        session.add_all(
            [
                Persona(
                    id=1,
                    ref_personality_id=101,
                    model="gpt4o",
                    population="pop_a",
                    description="Persona A",
                ),
                Persona(
                    id=2,
                    ref_personality_id=102,
                    model="gpt4o",
                    population="pop_b",
                    description="Persona B",
                ),
                Persona(
                    id=3,
                    ref_personality_id=103,
                    model="claude",
                    population="pop_a",
                    description="Persona C",
                ),
            ]
        )
        session.commit()

    personas = registration.fetch_personas(
        "gpt4o",
        ["pop_a", "pop_b", "pop_a"],
        session_factory=session_factory,
    )

    assert [persona.ref_personality_id for persona in personas] == [101, 102]


def test_register_questionnaire_experiments_persists_rows(
    session_factory, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(registration, "_get_repo_sha", lambda: "test-sha")
    monkeypatch.setattr(registration, "get_questionnaire_json", lambda _: "{}")
    monkeypatch.setattr(
        registration,
        "get_models_config",
        lambda _model: {
            "MODEL_PROVIDER": "openai",
            "MODEL_ID": "gpt-4o",
            "TEMPERATURE": 0.7,
            "TOP_P": 0.9,
        },
    )

    with session_factory() as session:
        session.add_all(
            [
                Persona(
                    id=1,
                    ref_personality_id=201,
                    model="gpt4o",
                    population="pop_a",
                    repetitions=2,
                    description="Persona repetition",
                ),
                Persona(
                    id=2,
                    ref_personality_id=202,
                    model="gpt4o",
                    population="pop_b",
                    repetitions=1,
                    description="Persona single",
                ),
                Persona(
                    id=3,
                    ref_personality_id=203,
                    model="claude",
                    population="pop_a",
                    repetitions=1,
                    description="Filtered out",
                ),
            ]
        )
        session.commit()

    group_handler = RecordingExperimentGroupHandler(session_factory)
    experiment_handler = RecordingExperimentHandler(session_factory)

    group_id, experiment_ids = registration.register_questionnaire_experiments(
        questionnaire="epqra",
        model="gpt4o",
        populations=["pop_a", "pop_b", "pop_a"],
        experiment_group_handler=group_handler,
        experiment_handler=experiment_handler,
        session_factory=session_factory,
    )

    assert group_id == 1
    assert experiment_ids == [1, 2, 3]

    with session_factory() as session:
        persisted_groups = session.exec(select(ExperimentsGroup)).all()
        persisted_experiments = session.exec(select(ExperimentsList)).all()

    assert len(persisted_groups) == 1
    assert len(persisted_experiments) == 3
    assert {exp.population for exp in persisted_experiments} == {"pop_a", "pop_b"}

    assert group_handler.calls == [
        _RecordingGroupCall(
            description="epqra questionnaire for gpt4o (pop_a, pop_b)",
            system_role=registration.SYSTEM_ROLE_TEMPLATE,
            base_prompt="{}",
            translated=False,
            temperature=0.7,
            top_p=0.9,
            ii_repeated=True,
        )
    ]

    populations_sequence = [call["population"] for call in experiment_handler.calls]
    assert populations_sequence == ["pop_a", "pop_a", "pop_b"]

    repetition_sequence = [call["repeated"] for call in experiment_handler.calls]
    assert repetition_sequence == [1, 2, 1]
