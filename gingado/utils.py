import sdmx
import datetime
import numpy as np
import pandas as pd
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
