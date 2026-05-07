import pytest
from pokertv.ocr import parse_amount


@pytest.mark.parametrize("raw,expected", [
    ("$12.50", 12.50),
    ("12.50", 12.50),
    ("$1,234.56", 1234.56),
    ("10BB", 10.0),
    ("10 BB", 10.0),
    ("0.05", 0.05),
    ("", 0.0),
    ("N/A", 0.0),
    ("--", 0.0),
    ("$0", 0.0),
    ("Call $4.00", 4.00),
    ("Raise to $12.50", 12.50),
])
def test_parse_amount(raw, expected):
    assert parse_amount(raw) == pytest.approx(expected)
