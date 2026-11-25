import importlib.util
from pathlib import Path

spec = importlib.util.spec_from_file_location("deploy", Path(__file__).with_name("deploy.py"))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

def test_plain_csv():
    assert abs(module.parse_probability("0.12345\n") - 0.12345) < 1e-12

def test_json_float():
    assert abs(module.parse_probability("0.25") - 0.25) < 1e-12

def test_csv_multi_value():
    assert abs(module.parse_probability("0.75,0.1") - 0.75) < 1e-12
