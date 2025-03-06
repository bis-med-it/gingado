from __future__ import annotations  # Allows multiple typing of arguments in Python versions prior to 3.10

import numpy as np
import pandas.api.types as ptypes
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, VotingClassifier, VotingRegressor
from sklearn.model_selection import GridSearchCV, ShuffleSplit, StratifiedShuffleSplit, TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.utils.metaestimators import available_if
from sklearn.utils.validation import check_is_fitted, validate_data

from .model_documentation import ModelCard, ggdModelDocumentation
from .utils import read_attr

__all__ = ['ggdBenchmark', 'ClassificationBenchmark', 'RegressionBenchmark']

def _benchmark_has(attr):
    # Check if the benchmark has certain attributes
    def check(self):
        getattr(self.benchmark, attr)
        return True
    return check
        
class ggdBenchmark(BaseEstimator):
    """The base class for gingado's Benchmark objects.

    This class provides the foundational functionality for benchmarking models, including
    setting up data splitters for time series data, fitting models, and comparing candidate models.
    """
    
    def _check_is_time_series(self, X, y=None):
        """
        Checks whether the data is a time series, and sets a data splitter
        accordingly if no data splitter is provided by the user
        Note: all data without an index (eg, a Numpy array) are considered to NOT be a time series
        """
        if hasattr(X, "index"):
            self.is_timeseries = (ptypes.is_datetime64_dtype(X.index)
                                  or ptypes.is_timedelta64_dtype(X.index))
        else:
            self.is_timeseries = False
        if self.is_timeseries and y is not None:
            if hasattr(y, "index"):
                self.is_timeseries = (ptypes.is_datetime64_dtype(y.index)
                                      or ptypes.is_timedelta64_dtype(y.index))
            else:
                self.is_timeseries = False

        if self.cv is None:
            self.cv = TimeSeriesSplit() if self.is_timeseries else self.default_cv

    def _creates_estimator(self):
        if self.estimator is None:
            pass

    def _fit(self, X, y):
        self._check_is_time_series(X, y)

        X, y = validate_data(X, y)

        if hasattr(self.estimator, "random_state"):
            self.estimator.random_state = self.random_state

        if self.param_search and self.param_grid:                
            self.benchmark = self.param_search(
                estimator=self.estimator, 
                param_grid=self.param_grid, 
                scoring=self.scoring, 
                cv=self.cv,
                verbose=self.verbose_grid)
        else:
            self.benchmark = self.estimator
            
        self.benchmark.fit(X, y)

        if self.auto_document is not None:
            self.document()

        return self
    
    def set_benchmark(
        self,
        estimator
    ):
        """Defines a fitted `estimator` as the new benchmark model.

        Args:
            estimator: A fitted estimator object.
        """
        check_is_fitted(estimator)
        self.benchmark = estimator

    def _read_candidate_params(self, candidates, ensemble_method):
                param_grid = []
                for i, model in enumerate(candidates):
                    check_is_fitted(model)
                    param_grid.append({
                        **{'candidate_estimator': [model]},
                        **{
                            'candidate_estimator__' + k: (v,)
                            for k, v in model.get_params().items()
                        }}
                    )
                if ensemble_method is not None:
                    candidate_models = [('candidate_'+str(i+1), model) for i, model in enumerate(candidates)]
                    voting = ensemble_method(estimators=candidate_models)
                    ensemble = {'candidate_estimator': [voting]}
                    param_grid.append(ensemble)
                return param_grid

    def compare(
        self,
        X:np.ndarray,
        y:np.ndarray,
        candidates,
        ensemble_method='object_default', 
        update_benchmark:bool=True
    ):
        """Compares the performance of the benchmark model with candidate models.

        Args:
            X: Input data of shape (n_samples, n_features).
            y: Target data of shape (n_samples,) or (n_samples, n_targets).
            candidates: Candidate estimator(s) for comparison.
            ensemble_method: Method to combine candidate estimators. Default is 'object_default'.
            update_benchmark: Whether to update the benchmark with the best performing model. Default is True.
        """

        check_is_fitted(self.benchmark)
        old_benchmark_params = self.benchmark.get_params()

        candidates = list(candidates) if type(candidates) != list else candidates
        list_candidates = [self.benchmark] + candidates
        
        est = self.benchmark.best_estimator_ if hasattr(self.benchmark, "best_estimator_") else self.benchmark
        cand_pipeline = Pipeline([('candidate_estimator', est)])
        
        if ensemble_method == 'object_default':
            ensemble_method = self.ensemble_method
        cand_params = self._read_candidate_params(list_candidates, ensemble_method=ensemble_method)
        cand_grid = GridSearchCV(cand_pipeline, cand_params, cv=self.cv, verbose=self.verbose_grid).fit(X, y)
        
        self.model_comparison_ = cand_grid

        if update_benchmark:
            if cand_grid.best_estimator_.get_params() != old_benchmark_params:
                self.set_benchmark(cand_grid)
                print("Benchmark updated!")
                print("New benchmark:")
                print(self.benchmark.best_estimator_)

        if self.auto_document is not None:
            self.document()

    def compare_fitted_candidates(self, X, y, candidates, scoring_func):
        check_is_fitted(self.benchmark)
        candidates = list(candidates) if type(candidates) != list else candidates
        for candidate in candidates:
            check_is_fitted(candidate)
        list_candidates = [self.benchmark] + candidates
        
        return {candidate.__repr__(): scoring_func(y, candidate.predict(X)) for candidate in list_candidates}

    def document(
        self, 
        documenter:ggdModelDocumentation|None=None
    ):
        """Documents the benchmark model using the specified template.

        Args:
            documenter: A gingado Documenter or the documenter set in `auto_document`. Default is None.
        """
        documenter = self.auto_document if documenter is None else documenter
        self.model_documentation = documenter()
        model_info = list(read_attr(self.benchmark))
        model_info = {k:v for i in model_info for k, v in i.items()}
        #self.model_documentation.read_model(self.benchmark)
        self.model_documentation.fill_model_info(model_info)

    @available_if(_benchmark_has("predict"))
    def predict(self, X, **predict_params):
        "Note: only available if the benchmark implements this method."
        return self.benchmark.predict(X, **predict_params)

    @available_if(_benchmark_has("fit_predict"))
    def fit_predict(self, X, y=None, **predict_params):
        "Note: only available if the benchmark implements this method."
        return self.benchmark.fit_predict(X, y, **predict_params)

    @available_if(_benchmark_has("predict_proba"))
    def predict_proba(self, X, **predict_proba_params):
        "Note: only available if the benchmark implements this method."
        return self.benchmark.predict_proba(X, **predict_proba_params)

    @available_if(_benchmark_has("decision_function"))
    def decision_function(self, X):
        "Note: only available if the benchmark implements this method."
        return self.benchmark.decision_function(X)

    @available_if(_benchmark_has("score"))
    def score(self, X):
        "Note: only available if the benchmark implements this method."
        return self.benchmark.score(X)

    @available_if(_benchmark_has("score_samples"))
    def score_samples(self, X):
        "Note: only available if the benchmark implements this method."
        return self.benchmark.score_samples(X)

    @available_if(_benchmark_has("predict_log_proba"))
    def predict_log_proba(self, X, **predict_log_proba_params):
        "Note: only available if the benchmark implements this method."
        return self.benchmark.predict_log_proba(X, **predict_log_proba_params)


class ClassificationBenchmark(ggdBenchmark, ClassifierMixin):
    "A gingado Benchmark object used for classification tasks"
    def __init__(self, 
    cv=None, 
    default_cv = StratifiedShuffleSplit(),
    estimator=RandomForestClassifier(oob_score=True), 
    param_grid={'n_estimators': [100, 250], 'max_features': ['sqrt', 'log2', None]}, 
    param_search=GridSearchCV, 
    scoring=None, 
    auto_document=ModelCard, 
    random_state=None,
    verbose_grid=False,
    ensemble_method=VotingClassifier):
        self.cv = cv
        self.default_cv = default_cv
        self.estimator = estimator
        self.param_grid = param_grid
        self.param_search = param_search
        self.scoring = scoring
        self.auto_document = auto_document
        self.random_state = random_state
        self.verbose_grid = verbose_grid
        self.ensemble_method = ensemble_method
        
    def fit(
        self, 
        X:np.ndarray,
        y:np.ndarray|None=None
    ):
        """
        Fit the ClassificationBenchmark model.

        Args:
            X (np.ndarray): Array-like data of shape (n_samples, n_features), representing the input data.
            y (np.ndarray, optional): Array-like data of shape (n_samples,) or (n_samples, n_targets), representing the target values. Defaults to None.
        
        Returns:
            ClassificationBenchmark: The instance of the model after fitting.
        """
        self._fit(X, y)
        return self


class RegressionBenchmark(ggdBenchmark, RegressorMixin):
    "A gingado Benchmark object used for regression tasks"
    def __init__(self, 
    cv=None, 
    default_cv=ShuffleSplit(),
    estimator=RandomForestRegressor(oob_score=True), 
    param_grid={'n_estimators': [100, 250], 'max_features': ['sqrt', 'log2', None]}, 
    param_search=GridSearchCV, 
    scoring=None, 
    auto_document=ModelCard, 
    random_state=None,
    verbose_grid=False,
    ensemble_method=VotingRegressor):
        self.cv = cv
        self.default_cv = default_cv
        self.estimator = estimator
        self.param_grid = param_grid
        self.param_search = param_search
        self.scoring = scoring
        self.auto_document = auto_document
        self.random_state = random_state
        self.verbose_grid = verbose_grid
        self.ensemble_method = ensemble_method

    def fit(
        self, 
        X:np.ndarray,
        y:np.ndarray|None=None
        ):
        """
        Fit the `RegressionBenchmark` model.

        Args:
            X (np.ndarray): Array-like data of shape (n_samples, n_features).
            y (np.ndarray | None, optional): Array-like data of shape (n_samples,) or (n_samples, n_targets) or None. Defaults to None.

        Returns:
            RegressionBenchmark: The instance of the model.
        """
        self._fit(X, y)
        return self
