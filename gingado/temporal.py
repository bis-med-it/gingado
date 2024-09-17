import pandas as pd
from sklearn.feature_selection import VarianceThreshold


def get_timefeat(
    df: pd.DataFrame, add_to_df: bool = True, drop_zero_variance: bool = True
) -> pd.DataFrame:
    """
    Generate temporal features from a DataFrame with a DatetimeIndex.

    This function creates various time-based features such as day of week,
    day of month, week of year, etc., based on the DatetimeIndex of the input DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame with a DatetimeIndex.
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

    dt_index = df.index
    time_features = pd.concat(
        [
            get_day_features(dt_index),
            get_week_features(dt_index),
            get_month_features(dt_index),
            get_quarter_features(dt_index),
        ],
        axis=1,
    )
    if drop_zero_variance:
        var_thresh = VarianceThreshold(threshold=0)
        var_thresh.set_output(transform="pandas")
        time_features = var_thresh.fit_transform(time_features)
    if add_to_df:
        return pd.concat([df, time_features], axis=1)
    return time_features


def get_day_features(dt_index: pd.DatetimeIndex) -> pd.DataFrame:
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


def get_week_features(dt_index: pd.DatetimeIndex) -> pd.DataFrame:
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


def get_month_features(dt_index: pd.DatetimeIndex) -> pd.DataFrame:
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


def get_quarter_features(dt_index: pd.DatetimeIndex) -> pd.DataFrame:
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
