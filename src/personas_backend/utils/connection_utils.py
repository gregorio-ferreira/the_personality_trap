"""Helper to bootstrap logging and database handlers for scripts."""

from typing import Any, Optional, Tuple

from personas_backend.db.db_handler import DatabaseHandler
from personas_backend.db.experiment_groups import ExperimentGroupHandler
from personas_backend.db.experiments import ExperimentHandler
from personas_backend.utils.logger import setup_logger


def setup_connections(
    logger_name: str = "experiment_runner",
) -> Tuple[Any, Optional[Any], ExperimentGroupHandler, ExperimentHandler]:
    """Initialize logger and database connections.

    Returns:
        Tuple containing: (logger, db_engine, exp_groups_handler, exp_handler)
    """
    logger = setup_logger(logger_name)
    db_handler = DatabaseHandler()
    db_engine = db_handler.connection  # SQLAlchemy Engine (may be None)
    exp_groups_handler = ExperimentGroupHandler(db_handler=db_handler, logger=logger)
    exp_handler = ExperimentHandler(db_handler=db_handler, logger=logger)
    return logger, db_engine, exp_groups_handler, exp_handler
