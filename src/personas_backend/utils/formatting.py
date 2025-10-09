"""Formatting helpers mirrored from the publication notebooks."""

import math
from typing import Union


def custom_format(value: Union[int, float, None]) -> str:
    """
    Format a numeric value with appropriate precision.

    Args:
        value: The numeric value to format

    Returns:
        Formatted string representation with appropriate precision
    """
    # Handle None, nan, or inf values
    if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
        return "{:.2f}".format(0)

    if value < 1e-99:
        return "{:.2f}".format(0)

    elif value < 0.01 and value > -0.01:
        return "{:.2E}".format(value)

    else:
        return "{:.2f}".format(value)
