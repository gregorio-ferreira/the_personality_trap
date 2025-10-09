from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Callable, Dict, Iterator

from sqlmodel import Session, select

from personas_backend.db.models import (
    EvalQuestionnaires,
    ExperimentRequestMetadata,
    ExperimentsGroup,
    ExperimentsList,
    Persona,
)
from personas_backend.db.experiments import ExperimentHandler
from personas_backend.evaluate_questionnaire import runner
from personas_backend.evaluate_questionnaire.runner import PendingExperiment, _process_single_experiment
from personas_backend.questionnaire.base_questionnaire import ExperimentRunResult


@dataclass
class _FakeQuestionnaireHandler:
    session_factory: Callable[[], Iterator[Session]]
    _next_eval_id: int = 1

    def process_experiment(
        self,
        my_experiment: Dict[str, int | str | None],
        persona_payload: Dict[str, object],
        *,
        system_role: str | None = None,
        auto_record: bool = False,
    ) -> ExperimentRunResult:
        with self.session_factory() as session:
            entry = EvalQuestionnaires(
                id=self._next_eval_id,
                experiment_id=my_experiment["experiment_id"],
                question_number=1,
                answer=1,
            )
            self._next_eval_id += 1
            session.add(entry)
            session.commit()
        return ExperimentRunResult(
            experiment_id=int(my_experiment["experiment_id"]),
            explanation="Synthetic explanation",
            request_json={"prompt": system_role or ""},
            response_json={"1": 1},
            request_metadata={"latency": 0.42},
        )


class _FakeDatabaseHandler:
    def __init__(self, engine):
        self.connection = engine

    def close_connection(self) -> None:  # pragma: no cover - compatibility shim
        self.connection.dispose()


def test_process_single_experiment_marks_success_and_persists_answers(
    session_factory, patch_get_session, sqlmodel_engine
) -> None:
    with session_factory() as session:
        persona = Persona(
            id=1,
            ref_personality_id=501,
            model="gpt4o",
            population="pop_a",
            description="Sample persona",
        )
        group = ExperimentsGroup(
            experiments_group_id=1,
            description="Test group",
            system_role="Describe <row.description>",
            base_prompt="{}",
            temperature=0.5,
            top_p=0.9,
        )
        experiment = ExperimentsList(
            experiment_id=1,
            experiments_group_id=1,
            questionnaire="epqra",
            language_instructions="english",
            language_questionnaire="english",
            model_provider="openai",
            model="gpt-4o",
            population="pop_a",
            personality_id=501,
            repeated=1,
        )
        session.add_all([persona, group, experiment])
        session.commit()

    with session_factory() as session:
        persona_row = session.get(Persona, 1)
        assert persona_row is not None
        persona_payload = runner._persona_to_payload(persona_row)

    pending = PendingExperiment(
        experiment_id=1,
        experiments_group_id=1,
        questionnaire="epqra",
        model_provider="openai",
        model_identifier="gpt-4o",
        model_name="gpt4o",
        population="pop_a",
        personality_id=501,
        repeated=1,
        system_role_template="Describe <row.description>",
        base_prompt="{}",
        group_description="Test group",
        persona_payload=persona_payload,
    )

    db_handler = _FakeDatabaseHandler(sqlmodel_engine)
    exp_handler = ExperimentHandler(db_handler=db_handler, logger=logging.getLogger("test"))
    questionnaire_handler = _FakeQuestionnaireHandler(session_factory=session_factory)

    result = _process_single_experiment(
        pending,
        questionnaire_handler=questionnaire_handler,
        exp_handler=exp_handler,
        engine=sqlmodel_engine,
        logger=logging.getLogger("runner-test"),
        schema="personality_trap",
    )

    assert isinstance(result, ExperimentRunResult)
    assert result.explanation == "Synthetic explanation"

    with session_factory() as session:
        updated_experiment = session.get(ExperimentsList, 1)
        assert updated_experiment is not None
        assert updated_experiment.succeeded is True
        assert updated_experiment.llm_explanation == "Synthetic explanation"

        eval_rows = session.exec(select(EvalQuestionnaires)).all()
        assert len(eval_rows) == 1
        assert eval_rows[0].experiment_id == 1

        metadata_rows = session.exec(select(ExperimentRequestMetadata)).all()
        assert len(metadata_rows) == 1
        assert metadata_rows[0].experiment_id == 1

        processed_group = session.get(ExperimentsGroup, 1)
        assert processed_group is not None
        assert processed_group.processed is True
