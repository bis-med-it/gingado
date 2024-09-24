import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_index_equal
from sklearn.dummy import DummyRegressor  # type: ignore
from sklearn.pipeline import Pipeline  # type: ignore

from gingado.temporal.transformer import TemporalFeatureTransformer
from gingado.temporal.types import (
    DayFeatures,
    Frequency,
    MonthFeatures,
    QuarterFeatures,
    WeekFeatures,
)


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
