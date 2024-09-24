"""Contains utilities to work with multi-frequency time series data."""

import pandas as pd
from sklearn.feature_selection import VarianceThreshold

from gingado.temporal.types import DateTimeLike, Frequency, FrequencyLike


def validate_and_get_freq(freq: FrequencyLike) -> Frequency:
    """Validate and return a pandas-compatible frequency string.

    This function checks if the provided frequency string is valid according to
    the Frequency enum and pandas' internal frequency validation. It converts
    the input to uppercase before validation.

    Args:
        freq (FrequencyLike): The input frequency to validate.

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
    df: pd.DataFrame | pd.Series,
    freq: FrequencyLike,
    add_to_df: bool = True,
    drop_zero_variance: bool = True,
) -> pd.DataFrame:
    """Generate temporal features from a DataFrame with a DatetimeIndex.

    This function creates various time-based features such as day of week,
    day of month, week of year, etc., based on the DatetimeIndex of the input DataFrame.

    Args:
        df (pd.DataFrame | pd.Series): Input DataFrame or Series with a DatetimeIndex.
        freq (FrequencyLike): Frequency of the input DataFrame
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
    """Calculate day-related temporal features from a DatetimeIndex.

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
    """Calculate week-related temporal features from a DatetimeIndex.

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
    """Calculate month-related temporal features from a DatetimeIndex.

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
    """Calculate quarter-related temporal features from a DatetimeIndex.

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


def _get_filtered_data(frame: pd.DataFrame, cutoff_date: DateTimeLike) -> pd.DataFrame | None:
    """Filter data up to a given date.

    Args:
        frame (pd.DataFrame): The input DataFrame to filter.
        cutoff_date (DateTimeLike): The cutoff date.

    Returns:
        Optional[pd.DataFrame]: Filtered DataFrame or None if empty.
    """
    filtered_data = frame.loc[frame.index <= cutoff_date]
    return filtered_data if not filtered_data.empty else None


def dates_Xy(
    X: dict[str, pd.DataFrame],
    y: pd.DataFrame | pd.Series,
    dates: list[DateTimeLike],
    freq_y: FrequencyLike = "MS",
) -> list[tuple[dict[str, pd.DataFrame | None], float]]:
    """Process time series data by creating snapshots for each date in the given list.

    Args:
        X (dict[str, pd.DataFrame]): Dictionary of DataFrames containing feature data at different
            frequencies.
        y (pd.DataFrame | pd.Series): DataFrame or Series containing target data.
        dates (list[DateTimeLike]): List of dates to process.
        freq_y (FrequencyLike): Frequency of the target data. Defaults to "M" (monthly).

    Returns:
        list[tuple[dict[str, pd.DataFrame | None], float]]: Processed data for each date.
    """
    result = []
    for date in dates:
        # data up to 'date' and y_value at 'date'
        X_filtered = {freq: _get_filtered_data(Xdata, date) for freq, Xdata in X.items()}
        y_value = y.loc[date]

        # Calculate temporal features for y data
        y_history = y.loc[:date]
        future_features = get_timefeat(y_history, freq=freq_y, add_to_df=False)
        X_filtered["future"] = future_features.iloc[-1:, :]  # Use only the last row

        result.append((X_filtered, y_value))

    return result
