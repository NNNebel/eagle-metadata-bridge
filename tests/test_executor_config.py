"""
test_executor_config.py

Unit tests for _load_config() and _config_to_settings() in executor.py,
and validation of the shipped config.json.
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

    def test_returns_empty_dict_on_invalid_json(self, monkeypatch, tmp_path, capsys):
        import executor
        (tmp_path / "config.json").write_text("NOT JSON", encoding="utf-8")
        monkeypatch.setattr(executor, "__file__", str(tmp_path / "executor.py"))
        result = _load_config()
        assert result == {}
        assert "ERROR" in capsys.readouterr().out

    def test_warns_on_unknown_top_level_key(self, monkeypatch, tmp_path, capsys):
        import executor
        (tmp_path / "config.json").write_text('{"unknown_key": 1}', encoding="utf-8")
        monkeypatch.setattr(executor, "__file__", str(tmp_path / "executor.py"))
        _load_config()
        assert "WARNING" in capsys.readouterr().out

    def test_errors_on_non_bool_value(self, monkeypatch, tmp_path, capsys):
        import executor
        cfg = {"tag": {"seed": "yes"}}
        (tmp_path / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
        monkeypatch.setattr(executor, "__file__", str(tmp_path / "executor.py"))
        result = _load_config()
        assert "ERROR" in capsys.readouterr().out
        # 不正な値は true に補正されている
        assert result["tag"]["seed"] is True

    def test_warns_on_unknown_section_key(self, monkeypatch, tmp_path, capsys):
        import executor
        cfg = {"tag": {"seed": False, "unknown_field": True}}
        (tmp_path / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
        monkeypatch.setattr(executor, "__file__", str(tmp_path / "executor.py"))
        _load_config()
        assert "WARNING" in capsys.readouterr().out


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


# ---------------------------------------------------------------------------
# Shipped config.json validation
# ---------------------------------------------------------------------------

_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config.json")
_VALID_SECTIONS = {"tag", "annotation"}


class TestShippedConfigJson:

    def _load(self):
        with open(_CONFIG_PATH, encoding="utf-8") as f:
            return json.load(f)

    def test_file_exists(self):
        assert os.path.isfile(_CONFIG_PATH), "config.json が見つかりません"

    def test_valid_json(self):
        # json.load が例外を出さなければ OK（test_file_exists に依存）
        data = self._load()
        assert isinstance(data, dict)

    def test_no_unknown_top_level_keys(self):
        data = self._load()
        allowed = _VALID_SECTIONS | {"eagle_port"}
        unknown = set(data.keys()) - allowed
        assert not unknown, f"未知のトップレベルキー: {unknown}"

    def test_tag_section_keys_are_valid(self):
        data = self._load()
        if "tag" not in data:
            pytest.skip("tag セクションなし")
        unknown = set(data["tag"].keys()) - set(_ALL_SETTING_KEYS)
        assert not unknown, f"tag セクションに未知のキー: {unknown}"

    def test_annotation_section_keys_are_valid(self):
        data = self._load()
        if "annotation" not in data:
            pytest.skip("annotation セクションなし")
        unknown = set(data["annotation"].keys()) - set(_ALL_SETTING_KEYS)
        assert not unknown, f"annotation セクションに未知のキー: {unknown}"

    def test_tag_values_are_boolean(self):
        data = self._load()
        if "tag" not in data:
            pytest.skip("tag セクションなし")
        for k, v in data["tag"].items():
            assert isinstance(v, bool), f"tag.{k} の値が bool でない: {v!r}"

    def test_annotation_values_are_boolean(self):
        data = self._load()
        if "annotation" not in data:
            pytest.skip("annotation セクションなし")
        for k, v in data["annotation"].items():
            assert isinstance(v, bool), f"annotation.{k} の値が bool でない: {v!r}"
