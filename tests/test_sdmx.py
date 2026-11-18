from types import SimpleNamespace
from unittest.mock import Mock

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from gingado.augmentation import AugmentSDMX
import gingado.utils as utils


@pytest.fixture
def sdmx_client(monkeypatch):
    client = Mock()
    client.dataflow.return_value = SimpleNamespace(
        dataflow={"FLOW": SimpleNamespace(_name="Test flow")}
    )
    monkeypatch.setattr(utils, "_get_sdmx_client", lambda source: client)
    return client


def test_load_sdmx_without_sources_returns_none(sdmx_client):
    assert utils.load_SDMX_data({}, {}, {}, verbose=False) is None
    sdmx_client.dataflow.assert_not_called()


def test_load_sdmx_without_matching_flows_returns_none(sdmx_client):
    result = utils.load_SDMX_data({"BIS": ["MISSING"]}, {}, {}, verbose=False)

    assert result is None
    sdmx_client.data.assert_not_called()


@pytest.mark.parametrize("flows", ["FLOW", ["FLOW"]])
def test_load_sdmx_matches_full_flow_codes(monkeypatch, sdmx_client, flows):
    sdmx_client.dataflow.return_value.dataflow = {
        name: SimpleNamespace(_name=name) for name in ["F", "LOW", "FLOW", "FLOW_EXTRA"]
    }
    monkeypatch.setattr(utils.sdmx, "to_pandas", Mock(return_value=pd.DataFrame()))

    result = utils.load_SDMX_data({"BIS": flows}, {}, {}, verbose=False)

    assert result is None
    sdmx_client.data.assert_called_once_with(
        "FLOW",
        key={},
        params={},
        headers={"Accept": "application/vnd.sdmx.genericdata+xml;version=2.1"},
    )


@pytest.mark.parametrize("flows", ["FLOW", ["FLOW"], "all"])
def test_load_sdmx_empty_results_return_none(monkeypatch, sdmx_client, flows):
    monkeypatch.setattr(utils.sdmx, "to_pandas", Mock(return_value=pd.DataFrame()))

    assert utils.load_SDMX_data({"BIS": flows}, {}, {}, verbose=False) is None
    sdmx_client.data.assert_called_once()


@pytest.mark.parametrize("flows", [["FLOW"], "all"])
def test_load_sdmx_failed_queries_return_none(sdmx_client, flows):
    sdmx_client.data.side_effect = ValueError("SDMX response could not be parsed")

    result = utils.load_SDMX_data({"BIS": flows}, {}, {}, verbose=False)

    assert result is None
    sdmx_client.data.assert_called_once()


@pytest.mark.parametrize("preceding_flow", [None, "failed", "empty"])
@pytest.mark.parametrize("source", ["BIS", "ECB"])
def test_load_sdmx_retains_successful_queries(
    monkeypatch, sdmx_client, preceding_flow, source
):
    index = pd.date_range("2020-01-01", periods=2, name="TIME_PERIOD")
    frame = pd.DataFrame(
        [1.0, 2.0],
        index=index,
        columns=pd.MultiIndex.from_tuples([("D", "US")]),
    )
    message = object()
    sources = {source: ["FLOW"]}
    converter = Mock(return_value=frame)
    if preceding_flow:
        sdmx_client.dataflow.return_value.dataflow = {
            "OTHER": SimpleNamespace(_name="Other flow"),
            "FLOW": SimpleNamespace(_name="Test flow"),
        }
        sources = {source: ["OTHER", "FLOW"]}
        if preceding_flow == "failed":
            sdmx_client.data.side_effect = [ValueError("Failed query"), message]
        else:
            sdmx_client.data.side_effect = [object(), message]
            converter.side_effect = [pd.DataFrame(), frame]
    else:
        sdmx_client.data.return_value = message
    monkeypatch.setattr(utils.sdmx, "to_pandas", converter)
    keys = {"FREQ": "D"}
    params = {"startPeriod": "2020-01-01"}

    result = utils.load_SDMX_data(sources, keys, params, verbose=False)

    expected = pd.DataFrame([1.0, 2.0], index=index, columns=[f"{source}__FLOW_D__US"])
    assert_frame_equal(result, expected)
    request_kwargs = {"key": keys, "params": params}
    if source == "BIS":
        request_kwargs["headers"] = {
            "Accept": "application/vnd.sdmx.genericdata+xml;version=2.1"
        }
    sdmx_client.data.assert_any_call("FLOW", **request_kwargs)
    converter.assert_called_with(message, datetime="TIME_PERIOD")
    assert converter.call_count == (2 if preceding_flow == "empty" else 1)


def test_augment_sdmx_defaults_to_current_bis_policy_flow():
    assert AugmentSDMX().sources == {"BIS": "WS_CBPOL"}