from datetime import datetime

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_index_equal
from sklearn.dummy import DummyRegressor  # type: ignore
from sklearn.pipeline import Pipeline  # type: ignore

from gingado.utils import get_timefeat, dates_Xy, TemporalFeatureTransformer
from gingado.internals import (
    DayFeatures,
    Frequency,
    InvalidTemporalFeature,
    MonthFeatures,
    QuarterFeatures,
    WeekFeatures,
)
import gingado.internals as tp


@pytest.fixture
def sample_df():
    """Fixture to create a sample DataFrame for testing."""
    date_range = pd.date_range(start="2021-01-01", end="2021-12-31", freq="D")
    df = pd.DataFrame(
        {
            "value": np.random.rand(len(date_range)),
        },
        index=date_range,
    )
    return df


@pytest.fixture(name="sample_df")
def sample_df_fixture() -> pd.DataFrame:
    """Create a sample DataFrame with a DatetimeIndex for testing."""

    date_range = pd.date_range(start="2023-01-01", end="2024-12-31", freq="D")
    return pd.DataFrame({"value": range(len(date_range))}, index=date_range)


@pytest.fixture(name="daily_df")
def daily_df_fixture() -> pd.DataFrame:
    """Create a sample DataFrame with daily frequency for testing."""

    date_range = pd.date_range(start="2023-01-01", end="2023-12-31", freq="D")
    return pd.DataFrame({"value": range(len(date_range))}, index=date_range)


@pytest.fixture(name="monthly_start_df")
def monthly_start_df_fixture() -> pd.DataFrame:
    """Create a sample DataFrame with monthly frequency for testing."""

    date_range = pd.date_range(start="2023-01-01", end="2023-12-31", freq="MS")
    return pd.DataFrame({"value": range(len(date_range))}, index=date_range)


@pytest.fixture(name="quarterly_start_df")
def quarterly_start_df_fixture() -> pd.DataFrame:
    """Create a sample DataFrame with quarterly frequency for testing."""

    date_range = pd.date_range(start="2023-01-01", end="2024-12-31", freq="QS")
    return pd.DataFrame({"value": range(len(date_range))}, index=date_range)


def test_frequency_enum_values():
    """Test that Frequency enum has correct values."""
    assert Frequency.DAILY.value == "D"
    assert Frequency.WEEKLY.value == "W"
    assert Frequency.MONTHLY.value == "MS"
    assert Frequency.QUARTERLY.value == "QS"


@pytest.mark.parametrize(
    "input_freq, expected",
    [
        ("D", Frequency.DAILY),
        ("w", Frequency.WEEKLY),
        ("MS", Frequency.MONTHLY),
        ("qs", Frequency.QUARTERLY),
    ],
)
def test_validate_and_get_freq_valid_inputs(input_freq, expected):
    """Test validate_and_get_freq with valid inputs."""
    assert tp.validate_and_get_freq(input_freq) == expected


@pytest.mark.parametrize("invalid_freq", ["INVALID", "DAY", "5"])
def test_validate_and_get_freq_invalid_inputs(invalid_freq):
    """Test validate_and_get_freq with invalid inputs."""
    with pytest.raises(ValueError):
        tp.validate_and_get_freq(invalid_freq)


def test_validate_and_get_freq_pandas_compatibility():
    """Test that returned frequencies are compatible with pandas."""
    for freq in Frequency:
        offset = pd.tseries.frequencies.to_offset(freq.value)
        assert offset is not None
        if freq == Frequency.WEEKLY:
            assert offset.name == "W-SUN"
        elif freq == Frequency.QUARTERLY:
            assert offset.name == "QS-JAN"
        else:
            assert offset.name == freq.value


@pytest.mark.parametrize(
    "frequency, num_columns",
    [
        (Frequency.DAILY, 12),
        (Frequency.WEEKLY, 8),
        (Frequency.MONTHLY, 5),
        (Frequency.QUARTERLY, 3),
    ],
)
def test_get_timefeat_add_to_df(
    sample_df: pd.DataFrame, frequency: Frequency, num_columns: int
) -> None:
    """Test get_timefeat function when add_to_df is True."""
    result = get_timefeat(sample_df, freq=frequency, add_to_df=True)
    assert (
        len(result.columns) == len(sample_df.columns) + num_columns
    )  # Original column + num_columns time features
    assert len(result) == len(sample_df)  # Same number of rows as before
    assert "value" in result.columns


@pytest.mark.parametrize(
    "frequency, num_columns",
    [
        (Frequency.DAILY, 12),
        (Frequency.WEEKLY, 8),
        (Frequency.MONTHLY, 5),
        (Frequency.QUARTERLY, 3),
    ],
)
def test_get_timefeat_return_only_features(
    sample_df: pd.DataFrame, frequency: Frequency, num_columns
) -> None:
    """Test get_timefeat function when add_to_df is False."""
    result = get_timefeat(sample_df, freq=frequency, add_to_df=False)
    assert len(result.columns) == num_columns  # Only time features
    assert len(result) == len(sample_df)  # Same number of rows as before
    assert "value" not in result.columns


@pytest.mark.parametrize("frequency", list(Frequency))
def test_get_timefeat_invalid_index(frequency: Frequency) -> None:
    """Test get_timefeat function with an invalid DataFrame index."""
    invalid_df = pd.DataFrame({"value": range(10)})
    with pytest.raises(ValueError):
        get_timefeat(invalid_df, freq=frequency)


def test_feature_values(sample_df: pd.DataFrame) -> None:
    """Test specific feature values generated by get_timefeat"""
    result = get_timefeat(sample_df, freq=Frequency.DAILY, add_to_df=False)
    assert result.loc["2023-01-01", "day_of_week"] == 6  # Sunday
    assert result.loc["2023-01-01", "day_of_month"] == 1
    assert result.loc["2023-01-01", "day_of_year"] == 1
    assert result.loc["2023-01-01", "week_of_month"] == 1
    assert result.loc["2023-01-01", "month_of_year"] == 1
    assert result.loc["2023-03-31", "quarter_end"] == 1
    assert result.loc["2023-12-31", "year_end"] == 1


def test_user_provided_features(sample_df: pd.DataFrame) -> None:
    """Tests that only user provided temporal feature are used"""
    result = get_timefeat(
        sample_df, freq=Frequency.DAILY, columns=["day_of_week", "month_of_year"], add_to_df=False
    )

    assert len(result.columns) == 2
    assert "day_of_week" in result.columns
    assert "month_of_year" in result.columns


def test_user_provided_wrong_frequency_features(monthly_start_df: pd.DataFrame) -> None:
    """Tests that a warning is issued for supported features for the wrong frequency"""
    with pytest.warns(
        UserWarning,
        match="Requested temporal feature day_of_week not available for data with frequency MS!",
    ):
        result = get_timefeat(
            monthly_start_df,
            freq=Frequency.MONTHLY,
            columns=["day_of_week", "month_of_year"],
            add_to_df=False,
        )

    assert result.columns == ["month_of_year"]


def test_user_provided_invalid_frequency(sample_df: pd.DataFrame) -> None:
    """Tests that an appropriate exception is thrown for invalid columns"""
    with pytest.raises(
        InvalidTemporalFeature, match="Invalid temporal feature passed: 'ace_of_spades'"
    ):
        get_timefeat(sample_df, freq=Frequency.DAILY, columns=["day_of_year", "ace_of_spades"])


def test_day_features(sample_df: pd.DataFrame) -> None:
    """Test the get_day_features function."""
    result = tp._get_day_features(pd.DatetimeIndex(sample_df.index))
    assert result.index.equals(sample_df.index)
    assert all([f.value in result.columns for f in DayFeatures])


def test_day_features_specific_values(sample_df: pd.DataFrame) -> None:
    """Test specific values for day features."""
    result = tp._get_day_features(pd.DatetimeIndex(sample_df.index))

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
    """Test the get_week_features function."""
    result = tp._get_week_features(pd.DatetimeIndex(sample_df.index))
    assert result.index.equals(sample_df.index)

    assert all([f.value in result.columns for f in WeekFeatures])


def test_week_features_specific_values(sample_df: pd.DataFrame) -> None:
    """Test specific values for week features."""
    result = tp._get_week_features(pd.DatetimeIndex(sample_df.index))

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
    """Test the get_month_features function."""
    result = tp._get_month_features(pd.DatetimeIndex(sample_df.index))
    assert result.index.equals(sample_df.index)
    assert all([f.value in result.columns for f in MonthFeatures])


def test_month_features_specific_values(sample_df: pd.DataFrame) -> None:
    """Test specific values for month features."""
    result = tp._get_month_features(pd.DatetimeIndex(sample_df.index))

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
    """Test the get_quarter_features function."""
    result = tp._get_quarter_features(pd.DatetimeIndex(sample_df.index))
    assert result.index.equals(sample_df.index)
    assert all([f.value in result.columns for f in QuarterFeatures])


def test_quarter_features_specific_values(sample_df: pd.DataFrame) -> None:
    """Test specific values for quarter features."""
    result = tp._get_quarter_features(pd.DatetimeIndex(sample_df.index))
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


@pytest.fixture(name="sample_data")
def sample_data_fixture():
    """Generate sample data for the dates_Xy method"""
    dates = pd.date_range(start="2023-01-01", end="2023-12-31", freq="D")
    X = {
        "d": pd.DataFrame({"feature1": np.random.rand(len(dates))}, index=dates),
        "w": pd.DataFrame({"feature2": np.random.rand(len(dates[::7]))}, index=dates[::7]),
    }
    y = pd.Series(np.random.rand(len(dates)), index=dates)
    return X, y


def test_dates_Xy_basic(sample_data):
    """Tests the basic structure returned by dates_Xy"""
    X, y = sample_data
    date_list = [datetime(2023, 3, 1), datetime(2023, 6, 1), datetime(2023, 9, 1)]
    result = dates_Xy(X, y, date_list)

    #
    assert isinstance(result, list)
    # The list has one entry for each date
    assert len(result) == len(date_list)
    for sample in result:
        # Each entry is a tuple containing X and y, where X is a dictionary and y a float
        assert isinstance(sample, tuple)
        assert len(sample) == 2
        X_sample, y_sample = sample
        assert isinstance(X_sample, dict)
        assert isinstance(y_sample, float)


def test_dates_Xy_filtered_data(sample_data):
    """Tests that filtering works as expected"""
    X, y = sample_data
    date_list = [datetime(2023, 3, 1), datetime(2023, 6, 1)]
    result = dates_Xy(X, y, date_list)

    assert len(result[0][0]["d"]) == 60  # 31 + 28 + 1 days
    assert len(result[1][0]["d"]) == 152  # 151 days from Jan to May and 1st of June


def test_dates_Xy_empty_X(sample_data):
    """Test the dates_Xy method when the input data is empty"""
    _, y = sample_data
    X = {}
    date_list = [datetime(2023, 3, 1)]
    expected = pd.DataFrame(
        index=pd.date_range(start="2023-03-01", end="2023-03-01", freq="D"),
        data={"month_of_quarter": 3, "month_of_year": 3},
    )
    result = dates_Xy(X, y, date_list)

    assert len(result) == 1
    assert list(result[0][0].keys()) == ["future"]
    assert_frame_equal(result[0][0]["future"], expected, check_dtype=False)


def test_dates_Xy_empty_date_list(sample_data):
    """Tests the dates_Xy method when an empty date list is passed"""
    X, y = sample_data
    date_list = []
    result = dates_Xy(X, y, date_list)

    assert not result


def test_dates_Xy_different_frequencies(sample_data):
    X, y = sample_data
    X["m"] = pd.DataFrame(
        {"feature3": np.random.rand(12)},
        index=pd.date_range(start="2023-01-01", end="2023-12-31", freq="MS"),
    )
    date_list = [datetime(2023, 6, 15)]
    result = dates_Xy(X, y, date_list)

    assert set(result[0][0].keys()) == {"d", "w", "m", "future"}
    assert len(result[0][0]["d"]) == 166  # Days from Jan 1 to Jun 15
    assert len(result[0][0]["w"]) == 24  # Weeks from Jan 1 to Jun 15
    assert len(result[0][0]["m"]) == 6  # Months from Jan to Jun


def test_dates_Xy_custom_freq_y(sample_data):
    X, y = sample_data
    date_list = [datetime(2023, 6, 1)]
    result = dates_Xy(X, y, date_list, freq_y="W")

    future_features = result[0][0]["future"]
    assert "week_of_month" in future_features.columns
    assert "week_of_quarter" in future_features.columns
    assert "week_of_year" in future_features.columns
    assert "quarter_end" in future_features.columns


def test_transformer_init():
    """Test the initialization of TemporalFeatureTransformer."""
    transformer = TemporalFeatureTransformer(freq="D")
    assert transformer.freq == Frequency.DAILY


def test_transformer_fit(sample_df):
    """Test the fit method of TemporalFeatureTransformer."""
    transformer = TemporalFeatureTransformer(freq="D")
    fitted_transformer = transformer.fit(sample_df)
    assert fitted_transformer is transformer


def test_transformer_transform(sample_df):
    """Test the transform method of TemporalFeatureTransformer."""
    transformer = TemporalFeatureTransformer(freq="D")
    transformed_df = transformer.transform(sample_df)

    assert isinstance(transformed_df, pd.DataFrame)
    assert_index_equal(transformed_df.index, sample_df.index)
    assert "value" in transformed_df.columns
    assert all([f.value in transformed_df.columns for f in DayFeatures])


def test_transformer_invalid_input():
    """Test the transformer with invalid input (no DatetimeIndex)."""
    df = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    transformer = TemporalFeatureTransformer(freq="D")

    with pytest.raises(ValueError, match="DataFrame index must be a DatetimeIndex"):
        transformer.transform(df)


def test_different_frequencies():
    """Test the transformer with different frequencies."""
    date_range = pd.date_range(start="2021-01-01", end="2021-12-31", freq="D")
    df = pd.DataFrame({"value": np.random.rand(len(date_range))}, index=date_range)

    transformer_daily = TemporalFeatureTransformer(freq="D")
    transformer_weekly = TemporalFeatureTransformer(freq="W")
    transformer_monthly = TemporalFeatureTransformer(freq="MS")

    transformed_daily = transformer_daily.transform(df)
    transformed_weekly = transformer_weekly.transform(df.resample("W").mean())
    transformed_monthly = transformer_monthly.transform(df.resample("MS").mean())

    assert "day_of_week" in transformed_daily.columns
    assert "week_of_year" in transformed_weekly.columns
    assert "month_of_year" in transformed_monthly.columns

    assert "day_of_week" not in transformed_monthly.columns
    assert "day_of_month" not in transformed_weekly.columns


def test_transformer_user_provided_args(sample_df):
    """Tests that the transformer uses only"""
    transformer = TemporalFeatureTransformer(
        freq="D", features=["day_of_month", "week_of_year", "month_of_quarter"]
    )
    df_transformed = transformer.transform(sample_df)
    assert sorted(df_transformed.columns) == [
        "day_of_month",
        "month_of_quarter",
        "value",
        "week_of_year",
    ]


def test_get_valid_features():
    """Test the get_valid_features static method of TemporalFeatureTransformer."""
    valid_features = TemporalFeatureTransformer.get_valid_features()

    # Check that the result is a dictionary
    assert isinstance(valid_features, dict), "get_valid_features should return a dictionary"

    # Check that the dictionary has the correct keys
    expected_keys = ["day_features", "week_features", "month_features", "quarter_features"]
    assert set(valid_features.keys()) == set(expected_keys)

    # Check that the lists contain the correct feature names
    assert set(valid_features["day_features"]) == set(feature.value for feature in DayFeatures)
    assert set(valid_features["week_features"]) == set(feature.value for feature in WeekFeatures)
    assert set(valid_features["month_features"]) == set(feature.value for feature in MonthFeatures)
    assert set(valid_features["quarter_features"]) == set(
        feature.value for feature in QuarterFeatures
    )


def test_pipeline():
    """Test that the transformer can be used in an sklearn pipeline"""
    # Create sample data
    dates = pd.date_range(start="2021-01-01", end="2021-12-31", freq="D")
    rng = np.random.default_rng(seed=42)
    df = pd.DataFrame(
        index=dates,
        data={
            "value": rng.uniform(size=len(dates)),
            "target": np.sin(np.arange(len(dates)) / 10) + rng.normal(0, 0.1, len(dates)),
        },
    )

    # Split the data
    X = df[["value"]]
    y = df["target"]

    # Create the pipeline
    pipeline = Pipeline(
        [
            ("temporal_features", TemporalFeatureTransformer(freq="D")),
            ("regressor", DummyRegressor(strategy="mean")),
        ]
    )

    # Fit the pipeline
    pipeline.fit(X, y)

    # Make predictions
    _ = pipeline.predict(X)

    # TODO: test feature names once correctly implemented
