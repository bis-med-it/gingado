import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin  # type: ignore
from sklearn.utils.validation import _check_feature_names_in

from gingado.temporal.temporal import get_timefeat, validate_and_get_freq
from gingado.temporal.types import (
    DayFeatures,
    FrequencyLike,
    MonthFeatures,
    QuarterFeatures,
    WeekFeatures,
)


class TemporalFeatureTransformer(BaseEstimator, TransformerMixin):
    """
    A scikit-learn transformer that adds temporal features to a pandas DataFrame.

    This transformer uses the get_timefeat function to generate temporal features
    based on the DatetimeIndex of the input DataFrame.
    """

    def __init__(
        self,
        freq: FrequencyLike,
        features: list[str] | None = None,
    ):
        """Configure and set up a temporal feature transformer.

        Args:
            freq (FrequencyLike): Frequency of the input DataFrame.
            features (list[str] | None): List of temporal features to include. If None, all temporal
                features are selected. Defaults to None.
        """
        self.freq = validate_and_get_freq(freq)
        self.features = features

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None):
        """Fit the transformer to the input DataFrame.

        This method doesn't actually do anything as the transformer doesn't need fitting.
        It's included to conform to the scikit-learn transformer interface.

        Args:
            X (pd.DataFrame): Input features.
            y (pd.Series | None): Target variable. Not used in this transformer.

        Returns:
            TemporalFeatureTransformer: The transformer instance.
        """
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform the input DataFrame by adding temporal features.

        Args:
            X (pd.DataFrame): Input features.

        Returns:
            pd.DataFrame | np.ndarray: Transformed DataFrame with added temporal features.

        Raises:
            ValueError: If the input DataFrame's index is not a DatetimeIndex.
        """
        X_transformed = get_timefeat(df=X, freq=self.freq, add_to_df=True)
        return X_transformed

    @staticmethod
    def get_valid_features() -> dict[str, list[str]]:
        """Get a dictionary of all valid features grouped by feature type."""
        return {
            "day_features": [f.value for f in DayFeatures],
            "week_features": [f.value for f in WeekFeatures],
            "month_features": [f.value for f in MonthFeatures],
            "quarter_features": [f.value for f in QuarterFeatures],
        }
