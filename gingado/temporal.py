from enum import Enum

import pandas as pd
from sklearn.feature_selection import VarianceThreshold


class Frequency(Enum):
    """
    Enum class representing valid frequency codes for time series data.

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


def validate_and_get_freq(freq: str | Frequency) -> Frequency:
    """
    Validate and return a pandas-compatible frequency string.

    This function checks if the provided frequency string is valid according to
    the Frequency enum and pandas' internal frequency validation. It converts
    the input to uppercase before validation.

    Args:
        freq_str (str | Frequency): The input frequency to validate.

    Returns:
        str: A valid pandas frequency string.

    Raises:
        ValueError: If the input frequency is not valid or not supported.
    """
    try:
        if isinstance(freq, str):
            freq = Frequency(freq.upper())
        # Additional check with pandas
        pd.tseries.frequencies.to_offset(freq.value)
        return freq
    except ValueError as exc:
        raise ValueError(
            f"Invalid frequency: {freq}. " f"Allowed values are {[f.value for f in Frequency]}"
        ) from exc


def get_timefeat(
    df: pd.DataFrame, freq: Frequency, add_to_df: bool = True, drop_zero_variance: bool = True
) -> pd.DataFrame:
    """
    Generate temporal features from a DataFrame with a DatetimeIndex.

    This function creates various time-based features such as day of week,
    day of month, week of year, etc., based on the DatetimeIndex of the input DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame with a DatetimeIndex.
        freq (Frequency): Frequency of the input DataFrame
        add_to_df (bool, optional): If True, append the generated features to the input DataFrame.
            If False, return only the generated features. Defaults to True.
        drop_zero_variance (bool, optional): If True, drop columns with zero variance. For example,
            the day of month feature is irrelevant for inputs with a monthly frequency.

    Returns:
        pd.DataFrame: A DataFrame containing the generated temporal features,
            either appended to the input DataFrame or as a separate DataFrame.

    Raises:
        ValueError: If the input DataFrame's index is not a DatetimeIndex.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("DataFrame index must be a DatetimeIndex")
    freq = validate_and_get_freq(freq)

    features = []

    if freq == Frequency.DAILY:
        features.append(_get_day_features(df.index))

    if freq in [Frequency.DAILY, Frequency.WEEKLY]:
        features.append(_get_week_features(df.index))

    if freq in [Frequency.DAILY, Frequency.WEEKLY, Frequency.MONTHLY]:
        features.append(_get_month_features(df.index))

    # We currently use these for all frequencies
    features.append(_get_quarter_features(df.index))

    time_features = pd.concat(features, axis=1) if features else pd.DataFrame(index=df.index)

    if drop_zero_variance:
        var_thresh = VarianceThreshold(threshold=0)
        var_thresh.set_output(transform="pandas")
        time_features = var_thresh.fit_transform(time_features)
    if add_to_df:
        return pd.concat([df, time_features], axis=1)
    return time_features


def _get_day_features(dt_index: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Calculate day-related temporal features from a DatetimeIndex.

    Args:
        dt_index (pd.DatetimeIndex): The DatetimeIndex to extract features from.

    Returns:
        pd.DataFrame: A DataFrame containing day-related features
    """
    return pd.DataFrame(
        index=dt_index,
        data={
            "day_of_week": dt_index.dayofweek,
            "day_of_month": dt_index.day,
            "day_of_quarter": dt_index.dayofyear - dt_index.to_period("Q").start_time.dayofyear + 1,
            "day_of_year": dt_index.dayofyear,
        },
    )


def _get_week_features(dt_index: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Calculate week-related temporal features from a DatetimeIndex.

    Args:
        dt_index (pd.DatetimeIndex): The DatetimeIndex to extract features from.

    Returns:
        pd.DataFrame: A DataFrame containing week-related features
    """
    return pd.DataFrame(
        index=dt_index,
        data={
            "week_of_month": dt_index.day.map(lambda x: (x - 1) // 7 + 1),
            "week_of_quarter": (dt_index.dayofyear - dt_index.to_period("Q").start_time.dayofyear)
            // 7
            + 1,
            "week_of_year": dt_index.isocalendar().week,
        },
    )


def _get_month_features(dt_index: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Calculate month-related temporal features from a DatetimeIndex.

    Args:
        dt_index (pd.DatetimeIndex): The DatetimeIndex to extract features from

    Returns:
        pd.DataFrame: A DataFrame containing month-related features
    """
    return pd.DataFrame(
        index=dt_index,
        data={
            "month_of_quarter": dt_index.month.map(lambda x: (x - 1) % 3 + 1),
            "month_of_year": dt_index.month,
        },
    )


def _get_quarter_features(dt_index: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Calculate quarter-related temporal features from a DatetimeIndex.

    Args:
        dt_index (pd.DatetimeIndex): The DatetimeIndex to extract features from.

    Returns:
        pd.DataFrame: A DataFrame containing quarter-related features
    """
    return pd.DataFrame(
        index=dt_index,
        data={
            "quarter_of_year": dt_index.quarter,
            "quarter_end": dt_index.is_quarter_end.astype(int),
            "year_end": dt_index.is_year_end.astype(int),
        },
    )
