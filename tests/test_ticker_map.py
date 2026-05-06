from pathlib import Path

from app.news.ticker_map import CompanyTickerMap, normalize_company_name, resolve_ticker


def test_normalize_company_name_handles_case_and_punctuation():
    assert normalize_company_name("NTPC Ltd.") == "ntpc ltd"
    assert normalize_company_name("Power-Grid, Corp") == "power grid corp"


def test_alias_to_ticker_resolution(tmp_path: Path):
    mapping_path = tmp_path / "company_tickers.json"
    mapping_path.write_text(
        """
        {
          "NTPC": {"ticker": "NTPC.NS", "aliases": ["NTPC Ltd", "NTPC Limited"]},
          "Power Grid": {"ticker": "POWERGRID.NS", "aliases": ["Power Grid Corp", "Powergrid"]}
        }
        """,
        encoding="utf-8",
    )

    ticker_map = CompanyTickerMap(mapping_path)
    assert ticker_map.resolve_ticker("ntpc limited") == "NTPC.NS"
    assert ticker_map.resolve_ticker("Powergrid") == "POWERGRID.NS"
    assert resolve_ticker("NTPC Ltd", mapping_path) == "NTPC.NS"
    assert ticker_map.resolve_ticker("Unknown Company") is None
