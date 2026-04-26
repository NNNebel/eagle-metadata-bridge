"""
test_executor_config.py

Unit tests for _load_config() and _config_to_settings() in executor.py.
"""
import json
import os
import tempfile
import pytest

from executor import _load_config, _config_to_settings, _ALL_SETTING_KEYS


class TestLoadConfig:

    def test_returns_empty_dict_when_no_file(self, monkeypatch, tmp_path):
        # Point executor module's __file__ to a directory with no config.json
        import executor
        monkeypatch.setattr(executor, "__file__", str(tmp_path / "executor.py"))
        assert _load_config() == {}

    def test_returns_parsed_json(self, monkeypatch, tmp_path):
        import executor
        config = {"tag": {"seed": False}, "annotation": {"negative": False}}
        (tmp_path / "config.json").write_text(json.dumps(config), encoding="utf-8")
        monkeypatch.setattr(executor, "__file__", str(tmp_path / "executor.py"))
        assert _load_config() == config

    def test_returns_empty_dict_on_invalid_json(self, monkeypatch, tmp_path):
        import executor
        (tmp_path / "config.json").write_text("NOT JSON", encoding="utf-8")
        monkeypatch.setattr(executor, "__file__", str(tmp_path / "executor.py"))
        assert _load_config() == {}


class TestConfigToSettings:

    def test_missing_section_returns_none(self):
        assert _config_to_settings({}, "tag") is None

    def test_empty_section_returns_none(self):
        assert _config_to_settings({"tag": {}}, "tag") is None

    def test_false_key_is_false(self):
        cfg = {"tag": {"seed": False, "cfg": False}}
        settings = _config_to_settings(cfg, "tag")
        assert settings["seed"] is False
        assert settings["cfg"] is False

    def test_omitted_keys_default_to_true(self):
        cfg = {"tag": {"seed": False}}
        settings = _config_to_settings(cfg, "tag")
        for k in _ALL_SETTING_KEYS:
            if k != "seed":
                assert settings[k] is True

    def test_all_keys_present_in_result(self):
        cfg = {"tag": {"seed": False}}
        settings = _config_to_settings(cfg, "tag")
        assert set(settings.keys()) == set(_ALL_SETTING_KEYS)

    def test_annotation_section(self):
        cfg = {"annotation": {"negative": False, "scheduler": False}}
        settings = _config_to_settings(cfg, "annotation")
        assert settings["negative"] is False
        assert settings["scheduler"] is False
        assert settings["checkpoint"] is True
