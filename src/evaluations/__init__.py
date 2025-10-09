"""Evaluation utilities for reproducing paper analyses.

This package provides clean, minimal utilities for replicating the
statistical analyses and tables from the research paper:

- **data_access**: Load personas, questionnaire scores, and experiment
  data from PostgreSQL
- **table_demographics**: Generate Tables 1-3 (demographic analysis with
  z-tests)
- **table_personality**: Generate Table 4 (EPQR-A personality scores with
  t-tests)
- **table_cronbach**: Generate Tables 6 and A5 (Cronbach's Alpha internal
  consistency analysis for EPQR-A and Big Five)
- **table_accuracy**: Generate Tables A6-A7 (accuracy and error metrics
  for EPQR-A in appendix)

Note: Table 5 (Pearson correlations) and Table A4 (Big Five scores) are
implemented directly in their respective notebooks using pandas/pingouin.

All functions are schema-aware and generate DataFrame outputs suitable for
publication tables or CSV export.
"""

from . import (  # noqa: F401
    data_access,
    table_accuracy,
    table_cronbach,
    table_demographics,
    table_personality,
)
