from src.inference.predict import _google_finance_symbol_candidates, _parse_google_finance_quote
from src.data.historical_loader import HistoricalLoader


def test_google_finance_quote_parser_extracts_price_and_time():
    sample = (
        "# Reliance Industries Ltd\n"
        "\n"
        "    ₹1,361.10 0.29%-3.90 Today Apr 20, 3:59:57 PM UTC+5:30 · INR · NSE · Disclaimer"
    )

    price, as_of_date, as_of_time, source = _parse_google_finance_quote(sample)

    assert price == 1361.10
    assert as_of_date == "Apr 20"
    assert "3:59:57 PM" in as_of_time
    assert source == "google_finance"


def test_google_finance_symbol_candidates_prioritize_india():
    assert _google_finance_symbol_candidates("RELIANCE")[:2] == ["RELIANCE:NSE", "RELIANCE:BOM"]
    assert _google_finance_symbol_candidates("RELIANCE.NS")[:2] == ["RELIANCE:NSE", "RELIANCE:BOM"]


def test_historical_loader_google_finance_candidates_prioritize_india(tmp_path):
    loader = HistoricalLoader(raw_data_dir=tmp_path)
    assert loader._google_finance_symbol_candidates("RELIANCE")[:2] == ["RELIANCE:NSE", "RELIANCE:BOM"]
    assert loader._google_finance_symbol_candidates("RELIANCE.BO")[:2] == ["RELIANCE:BOM", "RELIANCE:BSE"]


def test_historical_loader_parses_google_finance_snapshot(tmp_path):
    loader = HistoricalLoader(raw_data_dir=tmp_path)
    sample = """
    Reliance Industries Ltd
    ₹1,354.90
    Apr 21, 3:59:58 PM GMT+5:30 · INR · NSE
    Previous close
    The last closing price
    ₹1,363.30
    Day range
    The range between the high and low prices over the past day
    ₹1,350.10 - ₹1,369.80
    """

    parsed = loader._parse_google_finance_snapshot(sample)

    assert parsed["close"] == 1354.90
    assert parsed["open"] == 1363.30
    assert parsed["low"] == 1350.10
    assert parsed["high"] == 1369.80
    assert parsed["volume"] == 0.0
