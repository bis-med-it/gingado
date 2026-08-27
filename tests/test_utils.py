from gingado.utils import load_SDMX_data
import pandas as pd
import pytest
from unittest.mock import patch

import gingado.utils as utils


def test_load_sdmx_data_returns_none_without_dataflows():
    """No compatible SDMX data should leave the input unaugmented."""
    assert load_SDMX_data({}, {}, {}, verbose=False) is None


class _Dataflow:
    _name = "Test dataflow"


class _Dataflows:
    dataflow = {
        "WS_CBPOL": _Dataflow(),
        "WS_CBPOL_EXTRA": _Dataflow(),
    }


class _Client:
    def __init__(self):
        self.queried = []

    def dataflow(self):
        return _Dataflows()

    def data(self, dflow, key, params):
        self.queried.append(dflow)
        return dflow


def test_load_sdmx_data_matches_string_dataflow_exactly():
    """A string dataflow selection must not match similarly named flows."""
    client = _Client()
    with patch.object(utils, "_get_sdmx_client", return_value=client), patch.object(
        utils.sdmx,
        "to_pandas",
        return_value=pd.DataFrame(
            [[1]],
            columns=pd.MultiIndex.from_tuples([("series", "value")]),
            index=pd.DatetimeIndex(["2020-01-01"], name="TIME_PERIOD"),
        ),
    ):
        result = load_SDMX_data(
            {"BIS": "WS_CBPOL"},
            {"FREQ": "D"},
            {},
            verbose=False,
        )

    assert client.queried == ["WS_CBPOL"]
    assert list(result.columns) == ["BIS__WS_CBPOL_series__value"]


def test_load_sdmx_data_propagates_parser_errors():
    """Parser failures must not be misreported as missing observations."""
    client = _Client()

    def raise_parser_error(data, datetime):
        raise ValueError("parser failure")

    with patch.object(utils, "_get_sdmx_client", return_value=client), patch.object(
        utils.sdmx, "to_pandas", side_effect=raise_parser_error
    ):
        with pytest.raises(ValueError, match="parser failure"):
            load_SDMX_data({"BIS": "WS_CBPOL"}, {"FREQ": "D"}, {}, verbose=False)
