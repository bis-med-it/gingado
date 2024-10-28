import sdmx
import datetime
import numpy as np
import pandas as pd
from gingado.internals import DayFeatures, WeekFeatures, MonthFeatures, QuarterFeatures, DateTimeLike, Frequency, FrequencyLike, validate_and_get_freq, _check_valid_features, _get_day_features, _get_week_features, _get_month_features, _get_quarter_features
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

__all__ = ['get_datetime', 'read_attr', 'Lag', 'list_SDMX_sources', 'list_all_dataflows', 'load_SDMX_data']

def get_datetime():
    "Returns the time now"
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S %Z") 

def read_attr(
    obj
):
    """
    Reads and yields the type and values of fitted attributes from an object.
    
    Args:
        obj: Object from which attributes will be read.
    """
    for a in dir(obj):
        # if statement filters out non-interesting attributes
        if a == '_estimator_type' or (a.endswith("_") and not a.startswith("_") and not a.endswith("__")):
            try:
                model_attr = obj.__getattribute__(a)
                yield {a: model_attr}
            except:
                pass

class Lag(BaseEstimator, TransformerMixin):
    """
    A transformer for lagging variables.
    
    Args:
        lags (int): The number of lags to apply.
        jump (int): The number of initial observations to skip before applying the lag.
        keep_contemporaneous_X (bool): Whether to keep the contemporaneous values of X in the output.
    """
    def __init__(self, lags=1, jump=0, keep_contemporaneous_X=False):
        self.lags = lags
        self.jump = jump
        self.keep_contemporaneous_X = keep_contemporaneous_X
    
    def fit(
        self, 
        X:np.ndarray,
        y=None
    ):
        """
        Fits the Lag transformer.
        
        Args:
            X (np.ndarray): Array-like data of shape (n_samples, n_features).
            y: Array-like data of shape (n_samples,) or (n_samples, n_targets) or None.
            
        Returns:
            self: A fitted version of the `Lag` instance.
        """  
        self.index = None
        if hasattr(X, "index"):
            self.index = X.index
        else:
            if y is not None and hasattr(y, "index"):
                self.index = y.index
        X = self._validate_data(X)

        self.effective_lags_ = self.lags + self.jump
        return self

    def transform(
        self, 
        X:np.ndarray,
    ):
        """
        Applies the lag transformation to the dataset `X`.
        
        Args:
            X (np.ndarray): Array-like data of shape (n_samples, n_features).
            
        Returns:
            A lagged version of `X`.
        """
        X_forlag = X
        
        X = self._validate_data(X)
        check_is_fitted(self)
        X_lags = []
        X_colnames = list(self.feature_names_in_) if self.keep_contemporaneous_X else []
        for lag in range(self.effective_lags_):
            if lag < self.jump:
                continue
            lag_count = lag+1
            lag_X = np.roll(X_forlag, lag_count, axis=0)
            X_lags.append(lag_X)
            if hasattr(self, "feature_names_in_"):
                X_colnames = X_colnames + [col+"_lag_"+str(lag+1) for col in list(self.feature_names_in_)]
        X = np.concatenate(X_lags, axis=1)
        if self.keep_contemporaneous_X:
            X = np.concatenate([X_forlag, X], axis=1)
        X = X[self.effective_lags_:, :]
        if hasattr(self, "index") and self.index is not None:
            new_index = self.index[self.effective_lags_:]
            X = pd.DataFrame(X, index=new_index, columns=X_colnames)
        else:
            X = pd.DataFrame(X)
        return X

def list_SDMX_sources():
    """
    Fetches the list of SDMX sources.
    
    Returns:
        The list of codes representing the SDMX sources available for data download.
    """
    return sdmx.list_sources()

def list_all_dataflows(
    codes_only:bool=False,
    return_pandas:bool=True
):
    """
    Lists all SDMX dataflows. Note: When using as a parameter to an `AugmentSDMX` object
    or to the `load_SDMX_data` function, set `codes_only=True`"
    
    Args:
        codes_only (bool): Whether to return only the dataflow codes.
        return_pandas (bool): Whether to return the result in a pandas DataFrame format.
        
    Returns:
        All available dataflows for all SDMX sources.
    """
    sources = sdmx.list_sources()
    dflows = {}
    for src in sources:
        try:
            dflows[src] = sdmx.to_pandas(sdmx.Client(src).dataflow().dataflow)
            dflows[src] = dflows[src].index if codes_only else dflows[src].index.reset_index()
        except:
            pass
    if return_pandas:
        dflows = pd.concat({
            src: pd.DataFrame.from_dict(dflows)
            for src, dflows in dflows.items()
            })[0].rename('dataflow')
    return dflows

def load_SDMX_data(
    sources:dict,
    keys:dict,
    params:dict,
    verbose:bool=True
    ):
    """
    Loads datasets from SDMX.
    
    Args:
        sources (dict): A dictionary with the sources and dataflows per source.
        keys (dict): The keys to be used in the SDMX query.
        params (dict): The parameters to be used in the SDMX query.
        verbose (bool): Whether to communicate download steps to the user.
        
    Returns:
        A pandas DataFrame with data from SDMX or None if no data matches the sources, keys, and parameters.
    """
    data_sdmx = {}
    for source in list(sources.keys()):
        src_conn = sdmx.Client(source)
        src_dflows = src_conn.dataflow()
        if sources[source] == 'all':
            dflows = {k: v for k, v in src_dflows.dataflow.items()}
        else:
            dflows = {k: v for k, v in src_dflows.dataflow.items() if k in sources[source]}
        for dflow in list(dflows.keys()):
            if verbose: print(f"Querying data from {source}'s dataflow '{dflow}' - {dflows[dflow]._name}...")
            try:
                data = sdmx.to_pandas(src_conn.data(dflow, key=keys, params=params), datetime='TIME_PERIOD')
            except:
                if verbose: print("this dataflow does not have data in the desired frequency and time period.")
                continue
            data.columns = ['__'.join(col) for col in data.columns.to_flat_index()]
            data_sdmx[source+"__"+dflow] = data

    if len(data_sdmx.keys()) is None:
        return

    df = pd.concat(data_sdmx, axis=1)
    df.columns = ['_'.join(col) for col in df.columns.to_flat_index()]
    return df


def get_timefeat(
    df: pd.DataFrame | pd.Series,
    freq: FrequencyLike,
    columns: list[str] | None = None,
    add_to_df: bool = True,
) -> pd.DataFrame:
    """Generate temporal features from a DataFrame with a DatetimeIndex.

    This function creates various time-based features such as day of week,
    day of month, week of year, etc., based on the DatetimeIndex of the input DataFrame.

    Args:
        df (pd.DataFrame | pd.Series): Input DataFrame or Series with a DatetimeIndex.
        freq (FrequencyLike): Frequency of the input DataFrame. Can either be a string which is
            a supported pandas frequency alias or an gingado-interal Frequency object.
        columns (list[str], optional): List of colums with temporal feature names that should be
            kept. If None, all default temporal features are returned. Defaults to None.
        add_to_df (bool, optional): If True, append the generated features to the input DataFrame.
            If False, return only the generated features. Defaults to True.

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

    # Filter for user-provided features
    if columns is not None:
        valid_columns = _check_valid_features(columns, freq)
        time_features = time_features.loc[:, valid_columns]

    if add_to_df:
        return pd.concat([df, time_features], axis=1)
    return time_features


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
        X_transformed = get_timefeat(df=X, freq=self.freq, columns=self.features, add_to_df=True)
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
