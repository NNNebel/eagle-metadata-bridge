"""
test_bridge_fixtures.py

Fixture-based tests for executor.py graph traversal and annotation generation.
Uses real PNG/WebP images from comfyui-auto-tagger as fixtures (no synthetic data).

Cross-repo contract
-------------------
Fixture images (PNG/WebP) and core field expected values live in comfyui-auto-tagger.
Set COMFYUI_AUTO_TAGGER_PATH in CI to point to the checkout (see .github/workflows/test.yml).

Annotation expected values are Python-specific and live in tests/expected/ here
(annotation key only — core fields are verified against comfyui-auto-tagger expected).

Counterpart: comfyui-auto-tagger/tests/integration/eagle-bridge-fixtures.integration.test.js

To add a new fixture:
  1. Generate an image via ComfyUI with eagle-metadata-bridge node
  2. Copy PNG and WebP to comfyui-auto-tagger/tests/fixtures/bridge-<name>.{png,webp}
  3. Run analyze-image.js in comfyui-auto-tagger and commit the expected JSON
  4. Add a test case to TEST_CASES below
"""
import json
import os
import struct
import pytest

try:
    from PIL import Image
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False

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
EXPECTED_DIR = os.path.join(TESTS_DIR, 'expected')

_cat_path = os.environ.get(
    'COMFYUI_AUTO_TAGGER_PATH',
    os.path.join(TESTS_DIR, '..', '..', 'comfyui-auto-tagger')
)
CAT_FIXTURES_DIR = os.path.join(_cat_path, 'tests', 'fixtures')
CAT_EXPECTED_DIR = os.path.join(_cat_path, 'tests', 'expected')

# ---------------------------------------------------------------------------
# Image metadata extraction helpers
# ---------------------------------------------------------------------------

def _read_xmp_chunks(path):
    """Read XMP chunk from WebP RIFF container. Returns raw bytes or None."""
    with open(path, 'rb') as f:
        data = f.read()
    if data[:4] != b'RIFF' or data[8:12] != b'WEBP':
        return None
    offset = 12
    while offset < len(data) - 8:
        chunk_type = data[offset:offset+4]
        chunk_size = struct.unpack_from('<I', data, offset+4)[0]
        if chunk_type == b'XMP ':
            return data[offset+8:offset+8+chunk_size]
        offset += 8 + chunk_size + (chunk_size % 2)
    return None

def _parse_kv_metadata(text):
    """Parse 'key: {json}\\nkey2: {json}' format used in WebP XMP and WebP EXIF."""
    result = {}
    i = 0
    while i < len(text):
        colon = text.find(':', i)
        if colon < 0:
            break
        key = text[i:colon].strip().lower()
        rest = text[colon+1:].lstrip(' ')
        if rest.startswith('{'):
            # Find matching closing brace
            depth = 0
            end = colon + 1 + (len(text[colon+1:]) - len(rest))
            j = end
            in_str = False
            esc = False
            while j < len(text):
                c = text[j]
                if esc:
                    esc = False
                elif c == '\\' and in_str:
                    esc = True
                elif c == '"':
                    in_str = not in_str
                elif not in_str:
                    if c == '{':
                        depth += 1
                    elif c == '}':
                        depth -= 1
                        if depth == 0:
                            try:
                                result[key] = json.loads(text[end:j+1])
                            except json.JSONDecodeError:
                                pass
                            i = j + 1
                            break
                j += 1
            else:
                break
        else:
            i = colon + 1
    return result

def load_fixture_from_image(image_path):
    """
    Extract prompt and eagle_bridge metadata from a PNG or WebP image.
    Returns {'prompt': {...}, 'eagle_bridge': {...}} or None if unavailable.
    """
    if not os.path.exists(image_path):
        return None

    ext = os.path.splitext(image_path)[1].lower()

    if ext == '.png':
        if not _PIL_AVAILABLE:
            return None
        img = Image.open(image_path)
        text = img.text if hasattr(img, 'text') else {}
        prompt = json.loads(text['prompt']) if 'prompt' in text else None
        eagle_bridge = json.loads(text['eagle_bridge']) if 'eagle_bridge' in text else None
        if prompt is None or eagle_bridge is None:
            return None
        return {'prompt': prompt, 'eagle_bridge': eagle_bridge}

    if ext == '.webp':
        xmp_bytes = _read_xmp_chunks(image_path)
        if xmp_bytes is None:
            return None
        text = xmp_bytes.decode('utf-8', errors='replace')
        meta = _parse_kv_metadata(text)
        if 'prompt' not in meta or 'eagle_bridge' not in meta:
            return None
        return {'prompt': meta['prompt'], 'eagle_bridge': meta['eagle_bridge']}

    return None

# ---------------------------------------------------------------------------
# Expected loaders
# ---------------------------------------------------------------------------

def load_js_expected(name):
    """Load comfyui-auto-tagger expected (source of truth for core fields)."""
    path = os.path.join(CAT_EXPECTED_DIR, f'{name}.json')
    if not os.path.exists(path):
        return {}
    with open(path, encoding='utf-8') as f:
        return json.load(f)

def load_py_expected(name):
    """Load eagle-metadata-bridge expected (annotation only)."""
    path = os.path.join(EXPECTED_DIR, f'{name}.json')
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
        'fixture_png': 'bridge-simple.png',
        'fixture_webp': 'bridge-simple.webp',
        'label': 'single KSampler, novaAnimeXL',
        'check_fields': ['checkpoint', 'seed', 'steps', 'cfg', 'sampler', 'scheduler'],
        'check_loras': [],
        'check_base_positive_contains': None,
    },
    {
        'name': 'bridge-multi',
        'fixture_png': 'bridge-multi.png',
        'fixture_webp': 'bridge-multi.webp',
        'label': '2x KSampler, hassakuXL',
        'check_fields': ['checkpoint', 'seed', 'steps', 'cfg', 'sampler', 'scheduler'],
        'check_loras': [],
        'check_base_positive_contains': None,
    },
    {
        'name': 'bridge-conditioning-combine',
        'fixture_png': 'bridge-conditioning-combine.png',
        'fixture_webp': 'bridge-conditioning-combine.webp',
        'label': 'ImpactCombineConditionings in base sampler positive',
        'check_fields': ['checkpoint', 'seed', 'steps', 'cfg', 'sampler', 'scheduler'],
        'check_loras': [],
        'check_base_positive_contains': 'brown hair',
    },
]

# Expand into per-image test cases (PNG + WebP)
_PARAMETRIZE = []
for _c in TEST_CASES:
    for _fmt in ('png', 'webp'):
        _PARAMETRIZE.append({**_c, '_fmt': _fmt, '_id': f"{_c['name']}-{_fmt}"})

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_fixture(case):
    """Return loaded fixture dict or skip if unavailable."""
    fmt = case['_fmt']
    fname = case[f'fixture_{fmt}']
    path = os.path.join(CAT_FIXTURES_DIR, fname)
    data = load_fixture_from_image(path)
    if data is None:
        pytest.skip(f'fixture not available: {fname} (set COMFYUI_AUTO_TAGGER_PATH)')
    return data

def _get_meta(case):
    fixture = _get_fixture(case)
    return extract_metadata(fixture['prompt'], fixture['eagle_bridge']['final_node_id'])

# ---------------------------------------------------------------------------
# Parametrized tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('case', _PARAMETRIZE, ids=[c['_id'] for c in _PARAMETRIZE])
class TestBridgeFixtures:

    def test_fixture_image_exists(self, case):
        fmt = case['_fmt']
        path = os.path.join(CAT_FIXTURES_DIR, case[f'fixture_{fmt}'])
        if not os.path.exists(path):
            pytest.skip(f'fixture not available: {path}')
        assert os.path.exists(path)

    def test_extracts_without_error(self, case):
        assert isinstance(_get_meta(case), dict)

    def test_has_generation_steps(self, case):
        assert len(_get_meta(case).get('generation_steps', [])) > 0

    def test_base_sampler_is_first_step(self, case):
        steps = _get_meta(case).get('generation_steps', [])
        assert steps[0]['is_base'] is True

    def test_final_node_id_matches_js_expected(self, case):
        """eagle_bridge.final_node_id must match comfyui-auto-tagger expected."""
        js = load_js_expected(case['name'])
        if not js or 'eagle_bridge' not in js:
            pytest.skip('eagle_bridge not in js expected')
        fixture = _get_fixture(case)
        assert fixture['eagle_bridge']['final_node_id'] == js['eagle_bridge']['final_node_id']

    def test_core_fields_match_js_expected(self, case):
        """Core fields must match comfyui-auto-tagger (JS parser) expected values."""
        js = load_js_expected(case['name'])
        if not js:
            pytest.skip('comfyui-auto-tagger expected not available')
        meta = _get_meta(case)
        for field in case['check_fields']:
            if field in js:
                assert meta.get(field) == js[field], \
                    f"field '{field}': Python={meta.get(field)!r}, JS={js[field]!r}"

    def test_loras_match_js_expected(self, case):
        if not case['check_loras']:
            pytest.skip('no loras to check')
        js = load_js_expected(case['name'])
        if js and 'loras' in js:
            assert set(_get_meta(case).get('loras', [])) == set(js['loras'])
        else:
            loras = _get_meta(case).get('loras', [])
            for lora in case['check_loras']:
                assert lora in loras, f"expected lora: {lora}"

    def test_base_positive_contains(self, case):
        if not case['check_base_positive_contains']:
            pytest.skip('no positive text check')
        base = _get_meta(case)['generation_steps'][0]
        assert case['check_base_positive_contains'] in (base.get('positive') or '')

    def test_annotation_format(self, case):
        annotation = generate_annotation(_get_meta(case))
        assert annotation.startswith('[Generation Info]')
        assert '[Base Sampler -' in annotation
        assert 'Checkpoint:' in annotation

    def test_annotation_matches_expected(self, case):
        py_expected = load_py_expected(case['name'])
        if 'annotation' not in py_expected:
            pytest.skip('annotation not in py expected')
        assert generate_annotation(_get_meta(case)) == py_expected['annotation']

    def test_tags_include_checkpoint(self, case):
        meta = _get_meta(case)
        tags = generate_tags(meta)
        ckpt_no_ext = os.path.splitext(meta.get('checkpoint', ''))[0].lower()
        assert ckpt_no_ext in tags
