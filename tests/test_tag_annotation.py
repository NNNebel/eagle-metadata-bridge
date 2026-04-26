"""
test_tag_annotation.py

Unit tests for generate_tags() and generate_annotation().
These tests verify the Python implementations against known inputs,
independent of JS and cross-repo fixtures.

Coverage goals:
- generate_tags: checkpoint, lora, positive/negative tokenisation,
  parameter tags (seed/steps/cfg/sampler), deduplication
- generate_annotation: [Generation Info] header, Checkpoint/LoRA lines,
  per-step blocks (label, Seed, params line, Positive/Negative),
  CFG formatting, step label includes node ID,
  single-step checkpoint shown in block, empty-lora LoRA line
"""
import pytest
from metadata_parser.tag_generator import generate_tags
from metadata_parser.annotation import generate_annotation


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _simple_meta(overrides=None):
    base = {
        "checkpoint": "myModel_v10.safetensors",
        "loras": [],
        "generation_steps": [
            {
                "node_id": "5",
                "node_type": "KSampler",
                "is_base": True,
                "step_index": 1,
                "checkpoint": "myModel_v10.safetensors",
                "seed": 42,
                "steps": 20,
                "cfg": 7.0,
                "sampler": "euler",
                "scheduler": "normal",
                "positive": "1girl, simple background",
                "negative": "bad hands",
                "distance": 1,
            }
        ],
        "seed": 42,
        "steps": 20,
        "cfg": 7.0,
        "sampler": "euler",
        "scheduler": "normal",
        "positive": "1girl, simple background",
        "negative": "bad hands",
    }
    if overrides:
        base.update(overrides)
    return base


def _multi_meta():
    return {
        "checkpoint": "model.safetensors",
        "loras": ["loraA.safetensors"],
        "generation_steps": [
            {
                "node_id": "10",
                "node_type": "KSampler",
                "is_base": True,
                "step_index": 1,
                "checkpoint": "model.safetensors",
                "seed": 100,
                "steps": 30,
                "cfg": 8.0,
                "sampler": "dpmpp_2m",
                "scheduler": "simple",
                "positive": "masterpiece, 1girl",
                "negative": "worst quality",
                "distance": 2,
            },
            {
                "node_id": "20",
                "node_type": "KSampler",
                "is_base": False,
                "step_index": 2,
                "checkpoint": "model.safetensors",
                "seed": 200,
                "steps": 15,
                "cfg": 5.0,
                "sampler": "euler",
                "scheduler": "normal",
                "positive": "ultra detailed",
                "negative": "blurry",
                "distance": 1,
            },
        ],
        "seed": 100,
        "steps": 30,
        "cfg": 8.0,
        "sampler": "dpmpp_2m",
        "scheduler": "simple",
        "positive": "masterpiece, 1girl\nultra detailed",
        "negative": "worst quality\nblurry",
    }


# ---------------------------------------------------------------------------
# generate_tags
# ---------------------------------------------------------------------------

class TestGenerateTags:

    def test_checkpoint_tag(self):
        tags = generate_tags(_simple_meta())
        assert "mymodel_v10" in tags

    def test_lora_tag(self):
        meta = _simple_meta({"loras": ["style_lora_v2.safetensors"]})
        tags = generate_tags(meta)
        assert "style_lora_v2" in tags

    def test_positive_tokens(self):
        tags = generate_tags(_simple_meta())
        assert "1girl" in tags
        assert "simple background" in tags

    def test_negative_tokens_prefixed(self):
        tags = generate_tags(_simple_meta())
        assert "neg:bad hands" in tags

    def test_param_tags(self):
        tags = generate_tags(_simple_meta())
        assert "seed:42" in tags
        assert "steps:20" in tags
        assert "cfg:7.00" in tags
        assert "sampler:euler" in tags

    def test_cfg_two_decimal_places(self):
        meta = _simple_meta({"cfg": 8.0})
        meta["generation_steps"][0]["cfg"] = 8.0
        tags = generate_tags(meta)
        assert "cfg:8.00" in tags

    def test_attention_weight_stripped(self):
        meta = _simple_meta({"positive": "(1girl:1.2), [background]"})
        tags = generate_tags(meta)
        assert "1girl" in tags
        assert "background" in tags

    def test_empty_loras_no_lora_tag(self):
        tags = generate_tags(_simple_meta())
        assert not any(t.startswith("neg:") and "lora" in t for t in tags)


# ---------------------------------------------------------------------------
# generate_annotation
# ---------------------------------------------------------------------------

class TestGenerateAnnotation:

    def test_header(self):
        ann = generate_annotation(_simple_meta())
        assert ann.startswith("[Generation Info]")

    def test_checkpoint_line(self):
        ann = generate_annotation(_simple_meta())
        assert "Checkpoint: myModel_v10" in ann

    def test_lora_line_always_present(self):
        # Even with empty loras, LoRA line must appear
        ann = generate_annotation(_simple_meta())
        assert "LoRA: " in ann

    def test_lora_names_listed(self):
        meta = _simple_meta({"loras": ["styleA.safetensors", "styleB.safetensors"]})
        ann = generate_annotation(meta)
        assert "LoRA: styleA, styleB" in ann

    def test_step_label_includes_node_id(self):
        ann = generate_annotation(_simple_meta())
        assert "[Base Sampler - KSampler (ID: 5)]" in ann

    def test_step2_label(self):
        ann = generate_annotation(_multi_meta())
        assert "[Step 2 - KSampler (ID: 20)]" in ann

    def test_single_step_checkpoint_in_block(self):
        # Single-step: Checkpoint must appear both at top AND inside the step block
        ann = generate_annotation(_simple_meta())
        lines = ann.split("\n")
        ckpt_lines = [i for i, l in enumerate(lines) if "Checkpoint: myModel_v10" in l]
        assert len(ckpt_lines) == 2, f"Expected 2 Checkpoint lines, got {ckpt_lines}: {ann}"

    def test_multi_step_checkpoint_not_repeated_when_same(self):
        # Multi-step with same checkpoint: appears only at top, not in each step
        ann = generate_annotation(_multi_meta())
        lines = ann.split("\n")
        ckpt_lines = [l for l in lines if "Checkpoint: model" in l]
        assert len(ckpt_lines) == 1

    def test_cfg_one_decimal_place(self):
        ann = generate_annotation(_simple_meta())
        assert "CFG: 7.0" in ann
        assert "CFG: 7.00" not in ann

    def test_params_line_format(self):
        ann = generate_annotation(_simple_meta())
        assert "Steps: 20 | CFG: 7.0 | Sampler: euler | Scheduler: normal" in ann

    def test_seed_line(self):
        ann = generate_annotation(_simple_meta())
        assert "Seed: 42" in ann

    def test_positive_line(self):
        ann = generate_annotation(_simple_meta())
        assert "Positive: 1girl, simple background" in ann

    def test_negative_line(self):
        ann = generate_annotation(_simple_meta())
        assert "Negative: bad hands" in ann

    def test_multi_step_both_steps_present(self):
        ann = generate_annotation(_multi_meta())
        assert "[Base Sampler - KSampler (ID: 10)]" in ann
        assert "[Step 2 - KSampler (ID: 20)]" in ann
        assert "Seed: 100" in ann
        assert "Seed: 200" in ann

    # ------------------------------------------------------------------
    # Boundary values
    # ------------------------------------------------------------------

    def test_steps_zero_in_step(self):
        """steps=0 は有効な値として出力されること。"""
        meta = _simple_meta()
        meta["generation_steps"][0]["steps"] = 0
        ann = generate_annotation(meta)
        assert "Steps: 0" in ann

    def test_cfg_zero_in_step(self):
        """cfg=0 は有効な値として出力されること。"""
        meta = _simple_meta()
        meta["generation_steps"][0]["cfg"] = 0
        ann = generate_annotation(meta)
        assert "CFG: 0.0" in ann

    def test_steps_zero_in_fallback(self):
        """fallback パスでも steps=0 が出力されること。"""
        meta = {"generation_steps": [], "seed": 1, "steps": 0, "cfg": 7.0, "sampler": "euler"}
        ann = generate_annotation(meta)
        assert "Steps: 0" in ann

    def test_cfg_zero_in_fallback(self):
        """fallback パスでも cfg=0 が出力されること。"""
        meta = {"generation_steps": [], "seed": 1, "steps": 20, "cfg": 0, "sampler": "euler"}
        ann = generate_annotation(meta)
        assert "CFG: 0.0" in ann

    def test_checkpoint_path_stripped_in_annotation(self):
        """checkpoint にパスが含まれていても basename のみが出力されること。"""
        meta = _simple_meta({"checkpoint": "models/sd/myModel_v10.safetensors"})
        meta["generation_steps"][0]["checkpoint"] = "models/sd/myModel_v10.safetensors"
        ann = generate_annotation(meta)
        assert "Checkpoint: myModel_v10" in ann
        assert "models/" not in ann

    def test_blank_line_before_step_when_header_has_content(self):
        """ヘッダーに内容がある場合、最初のステップの前に空行が入ること。"""
        ann = generate_annotation(_simple_meta())
        lines = ann.split("\n")
        step_idx = next(i for i, l in enumerate(lines) if l.startswith("[Base Sampler"))
        assert lines[step_idx - 1] == ""

    def test_no_blank_line_before_step_when_header_empty(self):
        """checkpoint=off, lora=off の場合、ステップ前に余分な空行が入らないこと。"""
        ann = generate_annotation(_simple_meta(), {"checkpoint": False, "lora": False,
                                                    "seed": True, "steps": True, "cfg": True,
                                                    "sampler": True, "scheduler": True,
                                                    "positive": True, "negative": True})
        lines = ann.split("\n")
        assert lines[0] == "[Generation Info]"
        assert lines[1] != ""  # no blank line between header and first step

    # ------------------------------------------------------------------
    # Fallback positive/negative format
    # ------------------------------------------------------------------

    def test_no_generation_steps_fallback(self):
        meta = {
            "checkpoint": "model.safetensors",
            "loras": [],
            "generation_steps": [],
            "seed": 1,
            "steps": 10,
            "cfg": 5.0,
            "sampler": "euler",
            "scheduler": "normal",
            "positive": "test prompt",
            "negative": "bad quality",
        }
        ann = generate_annotation(meta)
        lines = ann.split("\n")
        # positive: blank line + [Positive Prompt] header + text
        pos_idx = lines.index("[Positive Prompt]")
        assert lines[pos_idx - 1] == ""
        assert lines[pos_idx + 1] == "test prompt"
        # negative: blank line + [Negative Prompt] header + text
        neg_idx = lines.index("[Negative Prompt]")
        assert lines[neg_idx - 1] == ""
        assert lines[neg_idx + 1] == "bad quality"

    def test_no_generation_steps_all_off_header_only(self):
        """フォールバックブロックで全フィールド off → ヘッダーだけ残ること（空行なし）。"""
        meta = {
            "checkpoint": "model.safetensors",
            "loras": [],
            "generation_steps": [],
            "seed": 1,
            "steps": 10,
            "cfg": 5.0,
            "sampler": "euler",
            "scheduler": "normal",
            "positive": "test",
            "negative": "bad",
        }
        all_off = {k: False for k in
                   ["checkpoint", "lora", "positive", "negative",
                    "seed", "steps", "cfg", "sampler", "scheduler"]}
        ann = generate_annotation(meta, all_off)
        assert ann == "[Generation Info]"


# ---------------------------------------------------------------------------
# Settings (customisation) tests
# ---------------------------------------------------------------------------

class TestTagGeneratorSettings:

    def test_default_settings_same_as_none(self):
        meta = _simple_meta()
        assert generate_tags(meta) == generate_tags(meta, None)

    def test_checkpoint_off(self):
        tags = generate_tags(_simple_meta(), {"checkpoint": False})
        assert not any("mymodel" in t for t in tags)

    def test_lora_off(self):
        meta = _simple_meta({"loras": ["myLora.safetensors"]})
        tags = generate_tags(meta, {"lora": False})
        assert not any("mylora" in t for t in tags)

    def test_positive_off(self):
        tags = generate_tags(_simple_meta(), {"positive": False})
        assert "1girl" not in tags
        assert "simple background" not in tags

    def test_negative_off(self):
        tags = generate_tags(_simple_meta(), {"negative": False})
        assert not any(t.startswith("neg:") for t in tags)

    def test_seed_off(self):
        tags = generate_tags(_simple_meta(), {"seed": False})
        assert not any(t.startswith("seed:") for t in tags)

    def test_cfg_off(self):
        tags = generate_tags(_simple_meta(), {"cfg": False})
        assert not any(t.startswith("cfg:") for t in tags)

    def test_all_off_empty(self):
        settings = {k: False for k in
                    ["checkpoint", "lora", "positive", "negative",
                     "seed", "steps", "cfg", "sampler", "scheduler"]}
        assert generate_tags(_simple_meta(), settings) == []


class TestAnnotationSettings:

    def test_default_settings_same_as_none(self):
        meta = _simple_meta()
        assert generate_annotation(meta) == generate_annotation(meta, None)

    def test_checkpoint_off(self):
        ann = generate_annotation(_simple_meta(), {"checkpoint": False})
        assert "Checkpoint:" not in ann

    def test_lora_off(self):
        ann = generate_annotation(_simple_meta(), {"lora": False})
        assert "LoRA:" not in ann

    def test_seed_off(self):
        ann = generate_annotation(_simple_meta(), {"seed": False})
        assert "Seed:" not in ann

    def test_cfg_off(self):
        ann = generate_annotation(_simple_meta(), {"cfg": False})
        assert "CFG:" not in ann

    def test_positive_off(self):
        ann = generate_annotation(_simple_meta(), {"positive": False})
        assert "Positive:" not in ann

    def test_negative_off(self):
        ann = generate_annotation(_simple_meta(), {"negative": False})
        assert "Negative:" not in ann

    def test_sampler_off(self):
        ann = generate_annotation(_simple_meta(), {"sampler": False})
        assert "Sampler:" not in ann

    def test_scheduler_off(self):
        ann = generate_annotation(_simple_meta(), {"scheduler": False})
        assert "Scheduler:" not in ann

    def test_all_off_no_step_label(self):
        """全フィールド off のとき [Base Sampler - ...] ラベルも出ないこと。"""
        all_off = {k: False for k in
                   ["checkpoint", "lora", "positive", "negative",
                    "seed", "steps", "cfg", "sampler", "scheduler"]}
        ann = generate_annotation(_simple_meta(), all_off)
        assert "[Base Sampler" not in ann
        assert "KSampler" not in ann

    def test_all_off_header_only(self):
        """全フィールド off のとき [Generation Info] ヘッダーだけ残ること。"""
        all_off = {k: False for k in
                   ["checkpoint", "lora", "positive", "negative",
                    "seed", "steps", "cfg", "sampler", "scheduler"]}
        ann = generate_annotation(_simple_meta(), all_off)
        assert ann == "[Generation Info]"


# ---------------------------------------------------------------------------
# Settings coverage — each settings key must produce output when ON and
# suppress it when OFF.  If a new key is added to _ALL_SETTING_KEYS but
# tag_generator / annotation doesn't implement it, the ON test will fail.
# ---------------------------------------------------------------------------

_TAG_COVERAGE_CASES = [
    # (key, meta_overrides, expected_tag_fragment)
    # generate_tags reads top-level fields (not generation_steps), so overrides
    # use top-level keys.  generation_steps is cleared to avoid noise from the
    # base fixture's positive/negative/seed/etc.
    ("checkpoint", {"checkpoint": "models/myModel.safetensors", "generation_steps": []},
     "mymodel"),
    ("lora",       {"loras": ["loras/myLora.safetensors"], "generation_steps": []},
     "mylora"),
    ("positive",   {"positive": "masterpiece", "generation_steps": []},
     "masterpiece"),
    ("negative",   {"negative": "bad quality", "generation_steps": []},
     "neg:bad quality"),
    ("seed",       {"seed": 99999, "generation_steps": []},
     "seed:99999"),
    ("steps",      {"steps": 77, "generation_steps": []},
     "steps:77"),
    ("cfg",        {"cfg": 3.5, "generation_steps": []},
     "cfg:3.50"),
    ("sampler",    {"sampler": "dpmpp_2m", "generation_steps": []},
     "sampler:dpmpp_2m"),
    ("scheduler",  {"scheduler": "karras", "generation_steps": []},
     "scheduler:karras"),
]


_STEP = {
    "node_id": "1", "node_type": "KSampler", "is_base": True,
    "step_index": 1, "distance": 1,
    "checkpoint": "myModel.safetensors",
    "seed": 99, "steps": 25, "cfg": 6.5,
    "sampler": "euler", "scheduler": "karras",
    "positive": "masterpiece", "negative": "bad quality",
}

_ANN_COVERAGE_CASES = [
    # (key, meta, expected_fragment)
    # generationSteps path
    ("checkpoint", {"checkpoint": "myModel.safetensors", "loras": [], "generation_steps": [_STEP]},
     "Checkpoint: myModel"),
    ("lora",       {"checkpoint": None, "loras": ["myLora.safetensors"], "generation_steps": [_STEP]},
     "LoRA: myLora"),
    ("seed",       {"checkpoint": None, "loras": [], "generation_steps": [_STEP]},
     "Seed: 99"),
    ("steps",      {"checkpoint": None, "loras": [], "generation_steps": [_STEP]},
     "Steps: 25"),
    ("cfg",        {"checkpoint": None, "loras": [], "generation_steps": [_STEP]},
     "CFG: 6.5"),
    ("sampler",    {"checkpoint": None, "loras": [], "generation_steps": [_STEP]},
     "Sampler: euler"),
    ("scheduler",  {"checkpoint": None, "loras": [], "generation_steps": [_STEP]},
     "Scheduler: karras"),
    ("positive",   {"checkpoint": None, "loras": [], "generation_steps": [_STEP]},
     "Positive: masterpiece"),
    ("negative",   {"checkpoint": None, "loras": [], "generation_steps": [_STEP]},
     "Negative: bad quality"),
    # fallback path (generation_steps empty)
    ("seed",       {"checkpoint": None, "loras": [], "generation_steps": [],
                    "seed": 99, "steps": 25, "cfg": 6.5, "sampler": "euler", "scheduler": "karras"},
     "Seed: 99"),
    ("scheduler",  {"checkpoint": None, "loras": [], "generation_steps": [],
                    "seed": 1, "scheduler": "karras"},
     "Scheduler: karras"),
    ("positive",   {"checkpoint": None, "loras": [], "generation_steps": [],
                    "seed": 1, "positive": "masterpiece"},
     "[Positive Prompt]"),
    ("negative",   {"checkpoint": None, "loras": [], "generation_steps": [],
                    "seed": 1, "negative": "bad quality"},
     "[Negative Prompt]"),
]

_ANN_COVERAGE_IDS = [
    f"{key}-{'step' if meta.get('generation_steps') else 'fallback'}"
    for key, meta, _ in _ANN_COVERAGE_CASES
]


class TestAnnotationSettingsCoverage:
    """Each settings key must produce expected text when ON and suppress it when OFF."""

    @pytest.mark.parametrize("key,meta,fragment", _ANN_COVERAGE_CASES, ids=_ANN_COVERAGE_IDS)
    def test_on_produces_output(self, key, meta, fragment):
        ann = generate_annotation(meta, {key: True})
        assert fragment in ann, \
            f"setting '{key}' ON: expected '{fragment}' in annotation, got:\n{ann}"

    @pytest.mark.parametrize("key,meta,fragment", _ANN_COVERAGE_CASES, ids=_ANN_COVERAGE_IDS)
    def test_off_suppresses_output(self, key, meta, fragment):
        ann = generate_annotation(meta, {key: False})
        assert fragment not in ann, \
            f"setting '{key}' OFF: unexpected '{fragment}' in annotation:\n{ann}"


class TestTagSettingsCoverage:
    """Each settings key must produce the expected tag when ON and omit it when OFF."""

    @pytest.mark.parametrize("key,meta_overrides,fragment", _TAG_COVERAGE_CASES,
                              ids=[c[0] for c in _TAG_COVERAGE_CASES])
    def test_on_produces_tag(self, key, meta_overrides, fragment):
        meta = {**_simple_meta(), **meta_overrides}
        tags = generate_tags(meta, {key: True})
        assert any(fragment in t for t in tags), \
            f"setting '{key}' ON: expected fragment '{fragment}' in tags, got {tags}"

    @pytest.mark.parametrize("key,meta_overrides,fragment", _TAG_COVERAGE_CASES,
                              ids=[c[0] for c in _TAG_COVERAGE_CASES])
    def test_off_suppresses_tag(self, key, meta_overrides, fragment):
        meta = {**_simple_meta(), **meta_overrides}
        tags = generate_tags(meta, {key: False})
        assert not any(fragment in t for t in tags), \
            f"setting '{key}' OFF: unexpected fragment '{fragment}' found in tags {tags}"
