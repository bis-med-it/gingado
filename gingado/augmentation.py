
from __future__ import annotations # allows multiple typing of arguments in Python versions prior to 3.10

import numpy as np
import pandas as pd

from .utils import load_SDMX_data
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, validate_data
from sklearn.feature_selection import VarianceThreshold

__all__ = ['AugmentSDMX']

class AugmentSDMX(BaseEstimator, TransformerMixin):
    """A transformer that augments a dataset using SDMX data.
    
    Attributes:
        sources (dict): A dictionary with sources as keys and dataflows as values.
        variance_threshold (float | None): Variables with lower variance through time are removed if specified. Otherwise, all variables are kept.
        propagate_last_known_value (bool): Whether to propagate the last known non-NA value to following dates.
        fillna (float | int): Value to use to fill missing data.
        verbose (bool): Whether to inform the user about the process progress.
    """
    
    InputIndexMessage = "The dataset to be augmented must have a row index with the date/time information"
    def _format_string(self):
        return "%Y-%m-%d" if self.data_freq_ == 'D' else "%Y-%m" if self.data_freq_ == 'M' else "%Y"
    
    def _get_dates(self):
        fstr = self._format_string()
        return {
            "startPeriod": min(self.index_).strftime(fstr),
            "endPeriod": max(self.index_).strftime(fstr),
        }

    def _transform(self, X, training=True):
        df = load_SDMX_data(sources=self.sources, keys=self.keys_, params=self.params_, verbose=self.verbose)
        if df is None:
            return X

        if training:
            # test that the dataset `X` has the same dimension as the one
            # used during training, which is an evidence they are the same
            n_samples_in_transform, n_features_in_transform = X.shape
            if n_samples_in_transform != self.n_samples_in_ or n_features_in_transform != self.n_features_in_:
                raise ValueError("The `X` passed to the transform() method must be compatible with the `X` used by the fit() method.")
            # during testing, we don't want the possibility of a different
            # set of columns being retained by virtue of different dynamics
            # in both datasets. For example, if a feature is included in the
            # training but during the test dates the variable didn't move, it
            # should not be subject to the test below so that it is still
            # included in the fitted data.
            self.features_stay_ = df.columns
            self.features_removed_ = None
            merge_df = True
            if self.variance_threshold:
                df_temp = df.dropna(axis=0, how='any')
                feat_sel = VarianceThreshold(threshold=self.variance_threshold)
                try:
                    feat_sel.fit(df)
                    self.features_stay_ = df.columns[feat_sel.get_support()]
                    self.features_removed_ = df.columns[~feat_sel.get_support()]
                    df = df.iloc[:, feat_sel.get_support()]
                    df.columns = feat_sel.get_feature_names_out()
                except ValueError as e:
                    print("No columns added to original data because " + str(e).lower())
                    merge_df = False
            df.dropna(axis=0, how='all', inplace=True)
            df.dropna(axis=1, how='all', inplace=True)  
            if merge_df:
                X = pd.merge(left=X, right=df, how='left', left_index=True, right_on='TIME_PERIOD')      
        
        if training is False: # if True, `X` would already have merged with `df` (or not if no added column)
            X = pd.merge(left=X, right=df, how='left', left_index=True, right_on='TIME_PERIOD')
        if 'TIME_PERIOD' in X.columns:
            X.drop(columns='TIME_PERIOD', inplace=True)
        if self.propagate_last_known_value:
            X.ffill(inplace=True)
        if self.fillna is not None:
            X.fillna(self.fillna)
        if training:
            X.index = self.index_
        return X

    def __init__(
        self,
        sources:dict={'BIS': 'WS_CBPOL_D'},
        variance_threshold:float|None=None,
        propagate_last_known_value:bool=True,
        fillna:float|int = 0,
        verbose:bool=True
        ):
        self.sources = sources
        self.variance_threshold = variance_threshold
        self.propagate_last_known_value = propagate_last_known_value
        self.fillna = fillna
        self.verbose = verbose

    def fit(
            self,
            X:pd.Series|pd.DataFrame,
            y:None=None
        ):
        """Fits the instance of AugmentSDMX to `X`, learning its time series frequency.

        Args:
            X (pd.Series | pd.DataFrame): Data having an index of `datetime` type.
            y (None): `y` is kept as an argument for API consistency only.

        Returns:
            AugmentSDMX: A fitted version of the same AugmentSDMX instance.
        """
        try:
            self.data_freq_ = X.index.to_series().diff().min().resolution_string
        except AttributeError:
            print(self.InputIndexMessage)
            raise
        self.index_ = X.index
        self.keys_ = {'FREQ': self.data_freq_}
        X = validate_data(X)

        # this variable below is only included to test for consistency \
        # if `fit` and `transform` are called separately with the same `X`
        self.n_samples_in_ = X.shape[0]

        return self

    def transform(
            self,
            X:pd.Series|pd.DataFrame,
            y:None=None,
            training:bool=False
        ) -> np.ndarray:
        """Transforms input dataset `X` by adding the requested data using SDMX.

        Args:
            X (pd.Series | pd.DataFrame): Data having an index of `datetime` type.
            y (None): `y` is kept as an argument for API consistency only.
            training (bool): `True` if `transform` is called during training, `False` (default) if called during testing.

        Returns:
            np.ndarray: `X` augmented with data from SDMX with the same number of samples but more columns.
        """
        check_is_fitted(self)
        self.params_ = self._get_dates()
        idx = X.index
        transf_X = self._transform(X, training=training)
        transf_X.index = idx
        return transf_X

    def fit_transform(
            self, 
            X:pd.Series|pd.DataFrame,
            y:None=None
            ) -> np.ndarray:
        """Fit to data, then transform it.

        Args:
            X (pd.Series | pd.DataFrame): Data having an index of `datetime` type.
            y (None): `y` is kept as an argument for API consistency only.

        Returns:
            np.ndarray: `X` augmented with data from SDMX with the same number of samples but more columns.
        """
        
        self.fit(X)
        self.params_ = self._get_dates()
        return self.transform(X, training=True)
