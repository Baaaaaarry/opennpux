import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]


def load_decoder():
    path = ROOT / "tools/models/decode_token_ids.py"
    spec = importlib.util.spec_from_file_location("decode_token_ids", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_parse_npu_token_ids():
    decoder = load_decoder()
    assert decoder.parse_token_ids("8160, 579,264") == [8160, 579, 264]


@pytest.mark.parametrize("value", ["", "1,", "-1", "4294967296", "text"])
def test_reject_invalid_npu_token_ids(value):
    decoder = load_decoder()
    with pytest.raises(ValueError):
        decoder.parse_token_ids(value)
