import pandas as pd
import pytest
import gingado.temporal as tp


@pytest.fixture(name="sample_df")
def sample_df_fixture() -> pd.DataFrame:
    """
    Create a sample DataFrame with a DatetimeIndex for testing.

    Returns:
        pd.DataFrame: A DataFrame with a single 'value' column and a DatetimeIndex
                      spanning from 2023-01-01 to 2024-12-31.
    """
    date_range = pd.date_range(start="2023-01-01", end="2024-12-31", freq="D")
    return pd.DataFrame({"value": range(len(date_range))}, index=date_range)


def test_get_timefeat_add_to_df(sample_df: pd.DataFrame) -> None:
    """
    Test get_timefeat function when add_to_df is True.

    Args:
        sample_df (pd.DataFrame): The sample DataFrame fixture.
    """
    result = tp.get_timefeat(sample_df, add_to_df=True)
    assert len(result.columns) == len(sample_df.columns) + 12  # Original column + 12 time features
    assert len(result) == len(sample_df)  # Same number of rows as before
    assert "value" in result.columns
    assert "day_of_week" in result.columns


def test_get_timefeat_return_only_features(sample_df: pd.DataFrame) -> None:
    """
    Test get_timefeat function when add_to_df is False.

    Args:
        sample_df (pd.DataFrame): The sample DataFrame fixture.
    """
    result = tp.get_timefeat(sample_df, add_to_df=False)
    assert len(result.columns) == 12  # Only time features
    assert "value" not in result.columns
    assert "day_of_week" in result.columns


def test_get_timefeat_invalid_index() -> None:
    """
    Test get_timefeat function with an invalid DataFrame index.
    """
    invalid_df = pd.DataFrame({"value": range(10)})
    with pytest.raises(ValueError):
        tp.get_timefeat(invalid_df)


def test_day_features(sample_df: pd.DataFrame) -> None:
    """
    Test the get_day_features function.

    Args:
        sample_df (pd.DataFrame): The sample DataFrame fixture.
    """
    result = tp.get_day_features(sample_df.index)
    assert result.index.equals(sample_df.index)
    assert "day_of_week" in result.columns
    assert "day_of_month" in result.columns
    assert "day_of_quarter" in result.columns
    assert "day_of_year" in result.columns


def test_day_features_specific_values(sample_df: pd.DataFrame) -> None:
    """
    Test specific values for day features.

    Args:
        sample_df (pd.DataFrame): The sample DataFrame fixture.
    """
    result = tp.get_day_features(sample_df.index)

    # Test for the first day of the year
    assert result.loc["2023-01-01", "day_of_week"] == 6  # Sunday
    assert result.loc["2023-01-01", "day_of_month"] == 1
    assert result.loc["2023-01-01", "day_of_quarter"] == 1
    assert result.loc["2023-01-01", "day_of_year"] == 1

    # Test for the last day of the year
    assert result.loc["2023-12-31", "day_of_week"] == 6  # Sunday
    assert result.loc["2023-12-31", "day_of_month"] == 31
    assert result.loc["2023-12-31", "day_of_quarter"] == 92
    assert result.loc["2023-12-31", "day_of_year"] == 365

    # Test for a leap year day
    assert result.loc["2024-02-29", "day_of_week"] == 3  # Thursday
    assert result.loc["2024-02-29", "day_of_month"] == 29
    assert result.loc["2024-02-29", "day_of_quarter"] == 60
    assert result.loc["2024-02-29", "day_of_year"] == 60

    # Test for the last day of a leap year
    assert result.loc["2024-12-31", "day_of_year"] == 366


def test_week_features(sample_df: pd.DataFrame) -> None:
    """
    Test the get_week_features function.

    Args:
        sample_df (pd.DataFrame): The sample DataFrame fixture.
    """
    result = tp.get_week_features(sample_df.index)
    assert result.index.equals(sample_df.index)

    assert "week_of_month" in result.columns
    assert "week_of_quarter" in result.columns
    assert "week_of_year" in result.columns


def test_week_features_specific_values(sample_df: pd.DataFrame) -> None:
    """
    Test specific values for week features.

    Args:
        sample_df (pd.DataFrame): The sample DataFrame fixture.
    """
    result = tp.get_week_features(sample_df.index)

    # Test for the first week of the year
    assert result.loc["2023-01-01", "week_of_month"] == 1
    assert result.loc["2023-01-01", "week_of_quarter"] == 1
    assert result.loc["2023-01-01", "week_of_year"] == 52  # Last week of previous year

    # Test for a mid-year week
    assert result.loc["2023-07-15", "week_of_month"] == 3
    assert result.loc["2023-07-15", "week_of_quarter"] == 3
    assert result.loc["2023-07-15", "week_of_year"] == 28

    # Test for the last week of the year
    assert result.loc["2023-12-31", "week_of_month"] == 5
    assert result.loc["2023-12-31", "week_of_quarter"] == 14
    assert result.loc["2023-12-31", "week_of_year"] == 52

    # Test for a leap year
    assert result.loc["2024-02-29", "week_of_month"] == 5
    assert result.loc["2024-02-29", "week_of_quarter"] == 9
    assert result.loc["2024-02-29", "week_of_year"] == 9


def test_month_features(sample_df: pd.DataFrame) -> None:
    """
    Test the get_month_features function.

    Args:
        sample_df (pd.DataFrame): The sample DataFrame fixture.
    """
    result = tp.get_month_features(sample_df.index)
    assert result.index.equals(sample_df.index)
    assert "month_of_quarter" in result.columns
    assert "month_of_year" in result.columns


def test_month_features_specific_values(sample_df: pd.DataFrame) -> None:
    """
    Test specific values for month features.

    Args:
        sample_df (pd.DataFrame): The sample DataFrame fixture.
    """
    result = tp.get_month_features(sample_df.index)

    # Test for January (first month of year and quarter)
    assert result.loc["2023-01-15", "month_of_quarter"] == 1
    assert result.loc["2023-01-15", "month_of_year"] == 1

    # Test for June (last month of Q2)
    assert result.loc["2023-06-15", "month_of_quarter"] == 3
    assert result.loc["2023-06-15", "month_of_year"] == 6

    # Test for December (last month of year and quarter)
    assert result.loc["2023-12-15", "month_of_quarter"] == 3
    assert result.loc["2023-12-15", "month_of_year"] == 12

    # Test for February in a leap year
    assert result.loc["2024-02-29", "month_of_quarter"] == 2
    assert result.loc["2024-02-29", "month_of_year"] == 2


def test_quarter_features(sample_df: pd.DataFrame) -> None:
    """
    Test the get_quarter_features function.

    Args:
        sample_df (pd.DataFrame): The sample DataFrame fixture.
    """
    result = tp.get_quarter_features(sample_df.index)
    assert result.index.equals(sample_df.index)
    assert "quarter_of_year" in result.columns
    assert "quarter_end" in result.columns
    assert "year_end" in result.columns


def test_quarter_features_specific_values(sample_df: pd.DataFrame) -> None:
    """
    Test specific values for quarter features.

    Args:
        sample_df (pd.DataFrame): The sample DataFrame fixture.
    """
    result = tp.get_quarter_features(sample_df.index)
    # Test for first day of Q1
    assert result.loc["2023-01-01", "quarter_of_year"] == 1
    assert result.loc["2023-01-01", "quarter_end"] == 0
    assert result.loc["2023-01-01", "year_end"] == 0

    # Test for last day of Q1
    assert result.loc["2023-03-31", "quarter_of_year"] == 1
    assert result.loc["2023-03-31", "quarter_end"] == 1
    assert result.loc["2023-03-31", "year_end"] == 0

    # Test for first day of Q4
    assert result.loc["2023-10-01", "quarter_of_year"] == 4
    assert result.loc["2023-10-01", "quarter_end"] == 0
    assert result.loc["2023-10-01", "year_end"] == 0

    # Test for last day of Q4 (also year end)
    assert result.loc["2023-12-31", "quarter_of_year"] == 4
    assert result.loc["2023-12-31", "quarter_end"] == 1
    assert result.loc["2023-12-31", "year_end"] == 1

    # Test for leap year
    assert result.loc["2024-02-29", "quarter_of_year"] == 1
    assert result.loc["2024-02-29", "quarter_end"] == 0
    assert result.loc["2024-02-29", "year_end"] == 0


def test_feature_values(sample_df: pd.DataFrame) -> None:
    """
    Test specific feature values generated by get_timefeat.

    Args:
        sample_df (pd.DataFrame): The sample DataFrame fixture.
    """
    result = tp.get_timefeat(sample_df, add_to_df=False)
    assert result.loc["2023-01-01", "day_of_week"] == 6  # Sunday
    assert result.loc["2023-01-01", "day_of_month"] == 1
    assert result.loc["2023-01-01", "day_of_year"] == 1
    assert result.loc["2023-01-01", "week_of_month"] == 1
    assert result.loc["2023-01-01", "month_of_year"] == 1
    assert result.loc["2023-03-31", "quarter_end"] == 1
    assert result.loc["2023-12-31", "year_end"] == 1
