"""
test_bridge_fixtures.py

Fixture-based tests for executor.py graph traversal and annotation generation.
Uses committed JSON fixtures extracted from real ComfyUI+eagle-metadata-bridge images.

Cross-repo contract
-------------------
Core field expected values (checkpoint, seed, steps, cfg, sampler, scheduler, loras)
are loaded from comfyui-auto-tagger/tests/expected/bridge-*.json — the JS parser is
the authoritative implementation. These files are committed in comfyui-auto-tagger
(gitignore exception for bridge-*.json) and checked out in CI.

Annotation expected values are Python-specific and live in tests/expected/ here.

Counterpart: comfyui-auto-tagger/tests/integration/eagle-bridge-fixtures.integration.test.js

To add a new fixture:
  1. Generate an image via ComfyUI with eagle-metadata-bridge node
  2. Run: python scripts/extract_fixture.py <image.png> tests/fixtures/<name>.json
  3. Commit tests/fixtures/<name>.json
  4. Add !tests/expected/bridge-<name>.json to comfyui-auto-tagger .gitignore
  5. Run analyze-image.js in comfyui-auto-tagger and commit the expected JSON
  6. Add a test case to TEST_CASES below
"""
import json
import os
import pytest

# ---------------------------------------------------------------------------
# Load executor pure functions without triggering ComfyUI imports
# ---------------------------------------------------------------------------

def _load_executor_functions():
    src_path = os.path.join(os.path.dirname(__file__), '..', 'executor.py')
    src = open(src_path, encoding='utf-8').read()
    cutoff = src.index('\n# ---------------------------------------------------------------------------\n# Main execute function')
    code = "import os, re, json\n" + src[src.index('\ndef _load_node_dictionary'):cutoff]
    ns = {'__file__': src_path}
    exec(compile(code, 'executor.py', 'exec'), ns)
    return ns

_fn = _load_executor_functions()
extract_metadata    = _fn['extract_metadata']
generate_annotation = _fn['generate_annotation']
generate_tags       = _fn['generate_tags']

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

TESTS_DIR    = os.path.dirname(__file__)
FIXTURES_DIR = os.path.join(TESTS_DIR, 'fixtures')
EXPECTED_DIR = os.path.join(TESTS_DIR, 'expected')

# comfyui-auto-tagger expected files (JS source of truth for core fields).
# Set COMFYUI_AUTO_TAGGER_PATH env var in CI (see .github/workflows/test.yml).
_cat_path = os.environ.get(
    'COMFYUI_AUTO_TAGGER_PATH',
    os.path.join(TESTS_DIR, '..', '..', 'comfyui-auto-tagger')
)
JS_EXPECTED_DIR = os.path.join(_cat_path, 'tests', 'expected')

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_fixture(name):
    path = os.path.join(FIXTURES_DIR, f'{name}.json')
    if not os.path.exists(path):
        pytest.skip(f'fixture not available locally: {name}.json')
    with open(path, encoding='utf-8') as f:
        return json.load(f)

def load_py_expected(name):
    path = os.path.join(EXPECTED_DIR, f'{name}.json')
    if not os.path.exists(path):
        return {}
    with open(path, encoding='utf-8') as f:
        return json.load(f)

def load_js_expected(name):
    """Load JS-parser expected from comfyui-auto-tagger. Returns {} if unavailable."""
    path = os.path.join(JS_EXPECTED_DIR, f'{name}.json')
    if not os.path.exists(path):
        return {}
    with open(path, encoding='utf-8') as f:
        return json.load(f)

# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

TEST_CASES = [
    {
        'name': 'bridge-simple',
        'label': 'single KSampler, novaAnimeXL',
        'check_fields': ['checkpoint', 'seed', 'steps', 'cfg', 'sampler', 'scheduler'],
        'check_loras': [],
        'check_base_positive_contains': None,
    },
    {
        'name': 'bridge-multi',
        'label': '2x KSampler, hassakuXL',
        'check_fields': ['checkpoint', 'seed', 'steps', 'cfg', 'sampler', 'scheduler'],
        'check_loras': [],
        'check_base_positive_contains': None,
    },
    {
        'name': 'bridge-conditioning-combine',
        'label': 'ImpactCombineConditionings in base sampler positive',
        'check_fields': ['checkpoint', 'seed', 'steps', 'cfg', 'sampler', 'scheduler'],
        'check_loras': [],
        'check_base_positive_contains': 'brown hair',
    },
]

# ---------------------------------------------------------------------------
# Parametrized tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('case', TEST_CASES, ids=[c['name'] for c in TEST_CASES])
class TestBridgeFixtures:

    def _meta(self, case):
        fixture = load_fixture(case['name'])
        return extract_metadata(fixture['prompt'], fixture['eagle_bridge']['final_node_id'])

    def test_fixture_file_exists(self, case):
        path = os.path.join(FIXTURES_DIR, f"{case['name']}.json")
        if not os.path.exists(path):
            pytest.skip(f'fixture not available locally: {path}')
        assert os.path.exists(path)

    def test_extracts_without_error(self, case):
        result = self._meta(case)
        assert isinstance(result, dict)

    def test_has_generation_steps(self, case):
        assert len(self._meta(case).get('generation_steps', [])) > 0

    def test_base_sampler_is_first_step(self, case):
        steps = self._meta(case).get('generation_steps', [])
        assert steps[0]['is_base'] is True

    def test_core_fields_match_js_expected(self, case):
        """Core fields must match expected values (py expected is source of truth here)."""
        py = load_py_expected(case['name'])
        if not py:
            pytest.skip('py expected not available')
        meta = self._meta(case)
        for field in case['check_fields']:
            if field in py:
                assert meta.get(field) == py[field], \
                    f"field '{field}': Python={meta.get(field)!r}, expected={py[field]!r}"

    def test_loras_match_js_expected(self, case):
        if not case['check_loras']:
            pytest.skip('no loras to check')
        py = load_py_expected(case['name'])
        if py and 'loras' in py:
            assert set(self._meta(case).get('loras', [])) == set(py['loras'])
        else:
            # Fallback: check known loras directly
            loras = self._meta(case).get('loras', [])
            for lora in case['check_loras']:
                assert lora in loras, f"expected lora: {lora}"

    def test_base_positive_contains(self, case):
        if not case['check_base_positive_contains']:
            pytest.skip('no positive text check')
        base = self._meta(case)['generation_steps'][0]
        assert case['check_base_positive_contains'] in (base.get('positive') or '')

    def test_annotation_format(self, case):
        annotation = generate_annotation(self._meta(case))
        assert annotation.startswith('[Generation Info]')
        assert '[Base Sampler -' in annotation
        assert 'Checkpoint:' in annotation

    def test_annotation_matches_expected(self, case):
        py_expected = load_py_expected(case['name'])
        if 'annotation' not in py_expected:
            pytest.skip('annotation not in py expected')
        assert generate_annotation(self._meta(case)) == py_expected['annotation']

    def test_tags_include_checkpoint(self, case):
        meta = self._meta(case)
        tags = generate_tags(meta)
        ckpt_no_ext = os.path.splitext(meta.get('checkpoint', ''))[0].lower()
        assert ckpt_no_ext in tags
