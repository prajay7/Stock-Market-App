from fastapi.testclient import TestClient

from app.main import app
import app.core.config as config_module


client = TestClient(app)


def test_health_endpoint():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_predict_validation_error_envelope():
    resp = client.post(
        "/predict",
        json={
            "symbols": [],
            "model_name": "hist_gb_classifier",
            "horizon_days": 1,
        },
    )
    assert resp.status_code == 422
    payload = resp.json()
    assert "error" in payload
    assert payload["error"]["code"] == "VALIDATION_ERROR"


def test_model_feature_importance_not_found_envelope():
    resp = client.get("/model-feature-importance", params={"model_name": "__missing_model__"})
    assert resp.status_code == 404
    payload = resp.json()
    assert "error" in payload
    assert payload["error"]["code"] == "FEATURE_IMPORTANCE_NOT_FOUND"


def test_model_feature_importance_endpoint(tmp_path, monkeypatch):
    model_dir = tmp_path / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = model_dir / "hist_gb_classifier_20260420120000.metadata.json"
    fi_path = model_dir / "hist_gb_classifier_20260420120000.feature_importance.json"

    metadata_path.write_text(
        "{"
        '"feature_list": ["f1", "f2"], '
        '"task_type": "classification", '
        '"target": "target_up_1d"'
        "}",
        encoding="utf-8",
    )
    fi_path.write_text(
        "{"
        '"feature_importance": ['
        '{"feature": "f1", "importance": 0.9},'
        '{"feature": "f2", "importance": 0.1}'
        "]"
        "}",
        encoding="utf-8",
    )

    test_settings = config_module.get_settings()
    monkeypatch.setattr(test_settings, "model_dir", model_dir)
    monkeypatch.setattr(config_module, "get_settings", lambda: test_settings)

    resp = client.get("/model-feature-importance", params={"model_name": "hist_gb_classifier", "top_k": 1})
    assert resp.status_code == 200
    payload = resp.json()
    assert payload["model_name"] == "hist_gb_classifier"
    assert payload["top_k"] == 1
    assert payload["feature_importance"][0]["feature"] == "f1"

    info_resp = client.get("/model-info", params={"model_name": "hist_gb_classifier"})
    assert info_resp.status_code == 200
    info_payload = info_resp.json()
    assert info_payload["feature_importance_top"][0]["feature"] == "f1"
