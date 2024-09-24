"""Contains type definitions for the temporal feature utilities"""

import datetime
from enum import Enum

import numpy as np
import pandas as pd


class Frequency(Enum):
    """Enum class representing valid frequency codes for time series data.

    These frequency codes align with pandas' frequency aliases and can be used
    for operations like resampling or date range generation.

    Attributes:
        DAILY (str): Daily frequency ('D')
        WEEKLY (str): Weekly frequency ('W')
        MONTHLY (str): Monthly start frequency ('MS')
        QUARTERLY (str): Quarterly start frequency ('QS')
    """

    DAILY = "D"
    WEEKLY = "W"
    MONTHLY = "MS"
    QUARTERLY = "QS"


FrequencyLike = str | Frequency
DateTimeLike = pd.Timestamp | datetime.datetime | np.datetime64 | str | datetime.date
