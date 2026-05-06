from pathlib import Path

from app.news.models import BeneficiaryCompany
from app.news.relations import CompanyRelations


def test_relationship_graph_loading_and_related_lookup(tmp_path: Path):
    relations_path = tmp_path / "company_relations.json"
    ticker_map_path = tmp_path / "company_tickers.json"

    relations_path.write_text(
        """
        {
          "NTPC": [
            {"company": "Power Grid", "relation": "transmission", "strength": 0.75},
            {"company": "BHEL", "relation": "equipment_supplier", "strength": 0.61}
          ]
        }
        """,
        encoding="utf-8",
    )
    ticker_map_path.write_text(
        """
        {
          "Power Grid": {"ticker": "POWERGRID.NS", "aliases": ["Powergrid"]},
          "BHEL": {"ticker": "BHEL.NS", "aliases": ["BHEL Ltd"]}
        }
        """,
        encoding="utf-8",
    )

    relations = CompanyRelations(relations_path, ticker_map_path)
    related = relations.related_companies("ntpc")
    assert len(related) == 2
    assert related[0]["ticker"] == "POWERGRID.NS"


def test_beneficiary_dedup_preserves_best_score():
    a = BeneficiaryCompany(company="Power Grid", relation="transmission", reason="graph", benefit_score=0.72)
    b = BeneficiaryCompany(company="power grid", relation="llm", reason="llm reason", benefit_score=0.81)
    c = BeneficiaryCompany(company="BHEL", relation="supplier", reason="graph", benefit_score=0.67)

    merged = CompanyRelations.merge_beneficiary_suggestions([a, c], [b])
    assert len(merged) == 2
    assert merged[0].company.lower() == "power grid"
    assert merged[0].benefit_score == 0.81
