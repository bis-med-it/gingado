import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin  # type: ignore

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
        drop_zero_variance: bool = True,
    ):
        """Configure and set up a temporal feature transformer.

        Args:
            freq (FrequencyLike): Frequency of the input DataFrame.
            features (list[str] | None): List of temporal features to include. If None, all temporal
                features are selected. Defaults to None.
            drop_zero_variance (bool): If True, drop columns with zero variance. Defaults to True.
        """
        self.freq = validate_and_get_freq(freq)
        self.drop_zero_variance = drop_zero_variance
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
        X_transformed = get_timefeat(
            df=X, freq=self.freq, add_to_df=True, drop_zero_variance=self.drop_zero_variance
        )
        # if self.features:

        return X_transformed

    def get_feature_names_out(self, input_features: list[str] | None = None) -> np.ndarray:
        if input_features is None and not hasattr(self, "X_"):
            raise AttributeError(
                "Transformer has not been fitted yet. Call 'fit' with appropriate arguments before"
                " using this method."
            )
        if input_features is None:
            input_features = self.X_.columns.tolist()
        # Generate a sample DataFrame to get the temporal feature names
        freq_str = validate_and_get_freq(self.freq).value
        sample_df = pd.DataFrame(index=pd.date_range(start="2021-01-01", periods=10, freq=freq_str))
        temporal_features = get_timefeat(
            sample_df, freq=self.freq, add_to_df=False, drop_zero_variance=self.drop_zero_variance
        )

        all_features = input_features + temporal_features.columns.tolist()
        return np.array(all_features)

    @staticmethod
    def get_valid_features() -> dict[str, list[str]]:
        """Get a dictionary of all valid features grouped by feature type."""
        return {
            "day_features": [f.value for f in DayFeatures],
            "week_features": [f.value for f in WeekFeatures],
            "month_features": [f.value for f in MonthFeatures],
            "quarter_features": [f.value for f in QuarterFeatures],
        }
