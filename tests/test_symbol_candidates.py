from src.data.historical_loader import HistoricalLoader


def test_internet_candidates_prioritize_indian_suffixes_for_base_symbol(tmp_path):
    loader = HistoricalLoader(raw_data_dir=tmp_path)
    candidates = loader._internet_symbol_candidates("RELIANCE")

    assert candidates[0] == "RELIANCE.NS"
    assert candidates[1] == "RELIANCE.BO"
    assert "RELIANCE" in candidates


def test_internet_candidates_preserve_existing_suffix_first(tmp_path):
    loader = HistoricalLoader(raw_data_dir=tmp_path)
    candidates = loader._internet_symbol_candidates("TCS.NS")

    assert candidates[0] == "TCS.NS"
