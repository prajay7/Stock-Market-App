from pathlib import Path

from app.news.relations import CompanyRelations


def test_aliases_resolve_company_and_ticker(tmp_path: Path):
    relations_path = tmp_path / "relations.json"
    alias_path = tmp_path / "ticker_aliases.json"

    relations_path.write_text(
        """
        {
          "Reliance Industries": {
            "ticker": "RELIANCE.NS",
            "aliases": ["RIL"],
            "relations": [
              {"company": "Jio Financial Services", "relation": "ecosystem", "strength": 0.6}
            ]
          }
        }
        """,
        encoding="utf-8",
    )
    alias_path.write_text(
        """
        {
          "companies": [
            {"company": "Reliance Industries", "ticker": "RELIANCE.NS", "aliases": ["Reliance", "RIL"]},
            {"company": "Jio Financial Services", "ticker": "JIOFIN.NS", "aliases": ["Jio Financial"]}
          ]
        }
        """,
        encoding="utf-8",
    )

    relations = CompanyRelations(relations_path, alias_path)

    assert relations.detect_primary_company("RIL plans a new expansion") == "Reliance Industries"
    assert relations.resolve_company("Reliance") == "Reliance Industries"
    assert relations.resolve_ticker("RIL") == "RELIANCE.NS"

    related = relations.related_companies("Reliance Industries")
    assert related
    assert related[0]["company"] == "Jio Financial Services"
    assert related[0]["ticker"] == "JIOFIN.NS"
