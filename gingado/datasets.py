from io import BytesIO
import os
from pathlib import Path
from zipfile import ZipFile

import pandas as pd
import requests
import numpy as np
from inspect import signature
from sklearn.utils import check_random_state

from gingado.internals import generate_timestamped_file_path, get_latest_timestamped_file_path
from gingado.settings import CACHE_DIRECTORY, CB_SPEECHES_CSV_BASE_FILENAME, CB_SPEECHES_BASE_URL, CB_SPEECHES_ZIP_BASE_FILENAME

__all__ = ['load_BarroLee_1994', 'make_causal_effect']

def load_BarroLee_1994(
    return_tuple:bool=True
):
    """Loads the dataset used in R. Barro and J.-W. Lee's "Sources of Economic Growth" (1994).

    Args:
        return_tuple (bool):  Whether to return the data in a tuple or jointly in a single pandas
                              DataFrame.

    Returns:
        pandas.DataFrame or tuple: If `return_tuple` is True, returns a tuple of (X, y), where `X` is a
        DataFrame of independent variables and `y` is a Series of the dependent variable. If False,
        returns a single DataFrame with both independent and dependent variables.

    """
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data', 'dataset_BarroLee_1994.csv')
    df = pd.read_csv(file_path)
    if return_tuple:
        y = df.pop('Outcome')
        X = df
        return X, y
    else:
        return df


def load_CB_speeches(
    year: str | int | list = 'all', cache: bool = True, timeout: float | None = 120, **kwargs
) -> pd.DataFrame:
    """Load Central Bankers speeches dataset from 
    Bank for International Settlements (2024). Central bank speeches, all years,
    https://www.bis.org/cbspeeches/download.htm.

    Args:
        year: Either 'all' to download all available central bank speeches or the year(s)
            to download. Defaults to 'all'.
        cache: If False, cached data will be ignored and the dataset will be downloaded again.
            Defaults to True.
        timeout: The timeout to for downloading each speeches file. Set to `None` to disable
            timeout. Defaults to 120.
        **kwargs. Additional keyword arguments which will be passed to pandas `read_csv` function.

    Returns:
        A pandas DataFrame containing the speeches dataset.

    Usage:
        >>> load_CB_speeches()

        >>> load_CB_speeches('2020')

        >>> load_CB_speeches([2020, 2021, 2022])
    """
    # Ensure year is list[str] for uniform handling
    if not isinstance(year, list):
        year = [str(year)]
    year = [str(y) for y in year]

    # Load data for each year
    cb_speeches_dfs = []
    for y in year:
        # Get expected filename (without extension) for speeches file
        filename_csv = (CB_SPEECHES_CSV_BASE_FILENAME if y == 'all' else f'{CB_SPEECHES_CSV_BASE_FILENAME}_{y}') + '.csv'

        # Get the file path of the CSV
        cb_speeches_file_path = str(Path(CACHE_DIRECTORY) / filename_csv)

        # Try to read the CSV file from cache
        cb_speeches_year_df: pd.DataFrame | None = None
        if cache:
            try:
                timestamped_file_path = get_latest_timestamped_file_path(cb_speeches_file_path)
                cb_speeches_year_df = pd.read_csv(timestamped_file_path, **kwargs)
            except FileNotFoundError:
                # File is not in cache
                cb_speeches_year_df = None

        # Download the CSV file, if it could not be loaded from cache
        if cb_speeches_year_df is None:
            # Get zip file URL
            filename_no_extension = (
                CB_SPEECHES_ZIP_BASE_FILENAME if y == 'all' else f'{CB_SPEECHES_ZIP_BASE_FILENAME}_{y}'
            )
            zip_url = CB_SPEECHES_BASE_URL + filename_no_extension + '.zip'

            # Download the zip file, unzip it and parse the CSV file:
            with requests.get(zip_url, timeout=timeout) as response:
                zip_file_content = response.content
                speeches_zip = ZipFile(BytesIO(zip_file_content))
                with speeches_zip.open(filename_no_extension + '.csv', 'r') as speeches_zip_file:
                    cb_speeches_year_df = pd.read_csv(speeches_zip_file, **kwargs)

            # Write file to cache
            timestamped_file_path = generate_timestamped_file_path(cb_speeches_file_path)
            Path(cb_speeches_file_path).parent.mkdir(exist_ok=True)  # Ensure parent dir exists
            cb_speeches_year_df.to_csv(timestamped_file_path, index=False)

        # Verify that the file in the cache is valid
        try:
            timestamped_file_path = get_latest_timestamped_file_path(cb_speeches_file_path)
            verify_df = pd.read_csv(timestamped_file_path)
            assert len(verify_df) > 0, f'CB speeches dataset at {timestamped_file_path} is empty.'
        except Exception as ex:
            raise RuntimeError('Verification error. See previous exception.') from ex

        # Add dataframe for year to aggregated list of dataframes
        cb_speeches_dfs.append(cb_speeches_year_df)

    # Concat all dataframes into single dataframe and return
    return pd.concat(cb_speeches_dfs)


def make_causal_effect(
    n_samples:int=100,
    n_features:int=100,
    pretreatment_outcome=lambda X, bias, rng: X[:, 1] + np.maximum(X[:, 2], 0) + bias + rng.standard_normal(size=X.shape[0]), 
    treatment_propensity=lambda X: 0.4 + 0.2 * (X[:, 0] > 0),
    treatment_assignment=lambda propensity, rng: rng.binomial(1, propensity),
    treatment=lambda assignment: assignment,
    treatment_effect=lambda treatment_value, X: np.maximum(X[:, 0], 0) * treatment_value,
    bias:float=0,
    noise:float=0,
    random_state=None,
    return_propensity:bool=False,
    return_assignment:bool=False,
    return_treatment_value:bool=False,
    return_treatment_effect:bool=True,
    return_pretreatment_y:bool=False,
    return_as_dict:bool=False 
):
    """Generates a simulated dataset to analyze causal effects of a treatment on an outcome variable.

    Args:
        n_samples (int): Number of observations in the dataset.
        n_features (int): Number of covariates for each observation.
        pretreatment_outcome (function): Function to generate outcome variable before any treatment.
        treatment_propensity (function or float): Function to generate treatment propensity or a fixed value for each observation.
        treatment_assignment (function): Function to determine treatment assignment based on propensity.
        treatment (function): Function to determine the magnitude of treatment for each treated observation.
        treatment_effect (function): Function to calculate the effect of treatment on the outcome variable.
        bias (float): Constant value added to the outcome variable.
        noise (float): Standard deviation of the noise added to the outcome variable. If 0, no noise is added.
        random_state (int, RandomState instance, or None): Seed or numpy random state instance for reproducibility.
        return_propensity (bool): If True, returns the treatment propensity for each observation.
        return_assignment (bool): If True, returns the treatment assignment status for each observation.
        return_treatment_value (bool): If True, returns the treatment value for each observation.
        return_treatment_effect (bool): If True, returns the treatment effect for each observation.
        return_pretreatment_y (bool): If True, returns the outcome variable of each observation before treatment effect.
        return_as_dict (bool): If True, returns the results as a dictionary; otherwise, returns as a list.
        
    Returns:
        A dictionary or list containing the simulated dataset components specified by the return flags.
    """
    generator = check_random_state(random_state)

    X = generator.standard_normal(size=(n_samples, n_features))

    if 'rng' in signature(pretreatment_outcome).parameters.keys():
        pretreatment_y = pretreatment_outcome(X=X, bias=bias, rng=generator)
    else:
        pretreatment_y = pretreatment_outcome(X=X, bias=bias)
    pretreatment_y = np.squeeze(pretreatment_y)
    if noise > 0.0:
        pretreatment_y += generator.normal(scale=noise, size=pretreatment_y.shape)

    # Since propensity may be a scalar (ie, the same propensity for all),
    # it is necessary to first check that it is callable.
    if callable(treatment_propensity):
        propensity = treatment_propensity(X=X)
    else:
        propensity = np.broadcast_to(treatment_propensity, pretreatment_y.shape)

    if 'rng' in signature(treatment_assignment).parameters.keys():
        assignment = treatment_assignment(propensity=propensity, rng=generator)
    else:
        assignment = treatment_assignment(propensity=propensity)

    # In case treatment is heterogenous amongst the treated observations,
    # the treatment function depends on `X`; otherwise only on `assignment`
    if 'X' in signature(treatment).parameters.keys():
        treatment_value = treatment(assignment=assignment, X=X)
    else:
        treatment_value = treatment(assignment=assignment)

    if len(treatment_value) == 1: treatment_value = treatment_value[0]

    # check that the treatment value is 0 for all observations that
    # are not assigned for treatment
    treatment_check = np.column_stack((assignment, treatment_value))
    if all(treatment_check[treatment_check[:, 0] == 0, 1] == 0) is False:
        raise ValueError("Argument `treatment` must be a function that returns 0 for observations with `assignment` == 0.\nOne suggestion is to multiply the desired treatment value with `assignment`.")

    # the code below checks whether the treatment effect responds to each unit's covariates
    # if not, then it just passes the treatment variable to `treatment_effect`
    if 'X' in signature(treatment_effect).parameters.keys(): 
        treat = treatment_effect(treatment_value=treatment_value, X=X)
    else:
        treat = treatment_effect(treatment_value=treatment_value)

    y = pretreatment_y + treat

    return_items = {'X': X, 'y': y}

    if return_propensity: return_items['propensity'] = propensity
    if return_assignment: return_items['treatment_assignment'] = assignment
    if return_treatment_value: return_items['treatment_value'] = treatment_value,
    if return_treatment_effect: return_items['treatment_effect'] = treat
    if return_pretreatment_y: return_items['pretreatment_y'] = pretreatment_y

    if return_as_dict == False:
        return_items = [v for k, v in return_items.items()]

    return return_items
