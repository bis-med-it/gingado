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


class DayFeatures(Enum):
    """Enum class representing day-based temporal features.

    These features provide various ways to represent a day within different
    time periods (week, month, quarter, year).

    Attributes:
        DAY_OF_WEEK (str): Day of the week (0-6, where 0 is Monday)
        DAY_OF_MONTH (str): Day of the month (1-31)
        DAY_OF_QUARTER (str): Day of the quarter (1-92)
        DAY_OF_YEAR (str): Day of the year (1-366)
    """

    DAY_OF_WEEK = "day_of_week"
    DAY_OF_MONTH = "day_of_month"
    DAY_OF_QUARTER = "day_of_quarter"
    DAY_OF_YEAR = "day_of_year"


class WeekFeatures(Enum):
    """Enum class representing week-based temporal features.

    These features provide various ways to represent a week within different
    time periods (month, quarter, year).

    Attributes:
        WEEK_OF_MONTH (str): Week of the month (1-5)
        WEEK_OF_QUARTER (str): Week of the quarter (1-13)
        WEEK_OF_YEAR (str): Week of the year (1-53)
    """

    WEEK_OF_MONTH = "week_of_month"
    WEEK_OF_QUARTER = "week_of_quarter"
    WEEK_OF_YEAR = "week_of_year"


class MonthFeatures(Enum):
    """Enum class representing month-based temporal features.

    These features provide ways to represent a month within a quarter and a year.

    Attributes:
        MONTH_OF_QUARTER (str): Month of the quarter (1-3)
        MONTH_OF_YEAR (str): Month of the year (1-12)
    """

    MONTH_OF_QUARTER = "month_of_quarter"
    MONTH_OF_YEAR = "month_of_year"


class QuarterFeatures(Enum):
    """Enum class representing quarter-based and year-end temporal features.

    These features provide ways to represent quarters within a year and
    indicate if a date is at the end of a quarter or year.

    Attributes:
        QUARTER_OF_YEAR (str): Quarter of the year (1-4)
        QUARTER_END (str): Boolean indicator for end of quarter (0 or 1)
        YEAR_END (str): Boolean indicator for end of year (0 or 1)
            Note: This is not strictly a quarterly feature but is included
            here for convenience in temporal feature generation.
    """

    QUARTER_OF_YEAR = "quarter_of_year"
    QUARTER_END = "quarter_end"
    YEAR_END = "year_end"
