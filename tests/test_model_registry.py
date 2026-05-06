import json

import joblib

from src.training.model_registry import ModelRegistry


def test_model_registry_writes_latest_and_prunes_old_versions(tmp_path):
    registry = ModelRegistry(tmp_path, keep_last_versions=2)
    versions = ["20260422110000", "20260422110100", "20260422110200"]

    for version in versions:
        registry.save_model({"version": version}, "xgboost_classifier", version)
        registry.save_metadata("xgboost_classifier", version, {"feature_list": ["f1"]})
        feature_path = tmp_path / f"xgboost_classifier_{version}.feature_importance.json"
        feature_path.write_text(json.dumps({"version": version, "feature_importance": []}), encoding="utf-8")

    deleted = registry.prune_old_versions("xgboost_classifier")

    assert (tmp_path / "xgboost_classifier_latest.joblib").exists()
    assert (tmp_path / "xgboost_classifier_latest.metadata.json").exists()
    assert joblib.load(tmp_path / "xgboost_classifier_latest.joblib")["version"] == versions[-1]

    assert not (tmp_path / "xgboost_classifier_20260422110000.joblib").exists()
    assert not (tmp_path / "xgboost_classifier_20260422110000.metadata.json").exists()
    assert not (tmp_path / "xgboost_classifier_20260422110000.feature_importance.json").exists()
    assert (tmp_path / "xgboost_classifier_20260422110100.joblib").exists()
    assert (tmp_path / "xgboost_classifier_20260422110200.joblib").exists()
    assert len(deleted) == 3


def test_model_registry_latest_only_mode_does_not_write_timestamped_versions(tmp_path):
    registry = ModelRegistry(tmp_path, keep_last_versions=0)
    version = "20260422110300"

    model_path = registry.save_model({"version": version}, "xgboost_classifier", version)
    metadata_path = registry.save_metadata("xgboost_classifier", version, {"feature_list": ["f1"]})

    assert model_path == tmp_path / "xgboost_classifier_latest.joblib"
    assert metadata_path == tmp_path / "xgboost_classifier_latest.metadata.json"
    assert (tmp_path / "xgboost_classifier_latest.joblib").exists()
    assert (tmp_path / "xgboost_classifier_latest.metadata.json").exists()
    assert not (tmp_path / f"xgboost_classifier_{version}.joblib").exists()
    assert not (tmp_path / f"xgboost_classifier_{version}.metadata.json").exists()
