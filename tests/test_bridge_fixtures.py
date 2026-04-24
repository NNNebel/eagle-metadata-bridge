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

from metadata_parser.comfyui_parser import extract_metadata
from metadata_parser.tag_generator import generate_tags
from metadata_parser.annotation import generate_annotation

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

TESTS_DIR = os.path.dirname(__file__)

_cat_path = os.environ.get(
    'COMFYUI_AUTO_TAGGER_PATH',
    os.path.join(TESTS_DIR, '..', '..', 'comfyui-auto-tagger')
)
CAT_FIXTURES_DIR = os.path.join(_cat_path, 'tests', 'fixtures')
CAT_EXPECTED_DIR = os.path.join(_cat_path, 'tests', 'expected')

# ---------------------------------------------------------------------------
# Image metadata extraction helpers
# ---------------------------------------------------------------------------

def _read_webp_metadata_chunk(path):
    """Read EXIF or XMP chunk from WebP RIFF container. Returns raw bytes or None."""
    with open(path, 'rb') as f:
        data = f.read()
    if data[:4] != b'RIFF' or data[8:12] != b'WEBP':
        return None
    offset = 12
    while offset < len(data) - 8:
        chunk_type = data[offset:offset+4]
        chunk_size = struct.unpack_from('<I', data, offset+4)[0]
        if chunk_type in (b'EXIF', b'XMP '):
            return data[offset+8:offset+8+chunk_size]
        offset += 8 + chunk_size + (chunk_size % 2)
    return None

def _parse_kv_metadata(text):
    """Parse 'key: {json}' entries from binary metadata text (EXIF or XMP)."""
    import re
    result = {}
    decoder = json.JSONDecoder()
    for key in ('workflow', 'prompt', 'eagle_bridge'):
        m = re.search(key + r':\s*(\{)', text, re.IGNORECASE)
        if m:
            try:
                obj, _ = decoder.raw_decode(text, m.start(1))
                result[key] = obj
            except json.JSONDecodeError:
                pass
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
        xmp_bytes = _read_webp_metadata_chunk(image_path)
        if xmp_bytes is None:
            return None
        text = xmp_bytes.decode('latin-1', errors='replace')
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


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

TEST_CASES = [
    {
        'name': 'bridge-simple',
        'fixture_png': 'bridge-simple.png',
        'fixture_webp': 'bridge-simple.webp',
        'label': 'single KSampler, novaAnimeXL',
    },
    {
        'name': 'bridge-multi',
        'fixture_png': 'bridge-multi.png',
        'fixture_webp': 'bridge-multi.webp',
        'label': '2x KSampler, hassakuXL',
    },
    {
        'name': 'bridge-conditioning-combine',
        'fixture_png': 'bridge-conditioning-combine.png',
        'fixture_webp': 'bridge-conditioning-combine.webp',
        'label': 'ImpactCombineConditionings in base sampler positive',
    },
    {
        'name': 'bridge-lora-simple',
        'fixture_png': 'bridge-lora-simple.png',
        'fixture_webp': 'bridge-lora-simple.webp',
        'label': 'LoraLoader single, novaAnimeXL',
    },
    {
        'name': 'bridge-lora-stack',
        'fixture_png': 'bridge-lora-stack.png',
        'fixture_webp': 'bridge-lora-stack.webp',
        'label': 'LoraLoaderStack rgthree, novaAnimeXL',
    },
    {
        'name': 'bridge-i2i',
        'fixture_png': 'bridge-i2i.png',
        'fixture_webp': 'bridge-i2i.webp',
        'label': 'DetailerForEachDebug (ADetailer), novaAnimeXL',
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
        """All top-level fields must match comfyui-auto-tagger (JS parser) expected values."""
        js = load_js_expected(case['name'])
        if not js:
            pytest.skip('comfyui-auto-tagger expected not available')
        meta = _get_meta(case)
        scalar_fields = ['checkpoint', 'seed', 'steps', 'cfg', 'sampler', 'scheduler',
                         'positive', 'negative']
        for field in scalar_fields:
            if field in js:
                assert meta.get(field) == js[field], \
                    f"field '{field}': Python={meta.get(field)!r}, JS={js[field]!r}"
        if 'loras' in js:
            assert set(meta.get('loras', [])) == set(js['loras']), \
                f"loras: Python={meta.get('loras')!r}, JS={js['loras']!r}"

    def test_generation_steps_match_js_expected(self, case):
        """Each generation step must match comfyui-auto-tagger (JS) step data field-by-field."""
        js = load_js_expected(case['name'])
        if not js or 'generationSteps' not in js:
            pytest.skip('generationSteps not in js expected')
        meta = _get_meta(case)
        py_steps = meta.get('generation_steps', [])
        js_steps = js['generationSteps']
        assert len(py_steps) == len(js_steps), \
            f"step count: Python={len(py_steps)}, JS={len(js_steps)}"
        # camelCase → snake_case field map (only fields Python produces)
        field_map = [
            ('seed',       'seed'),
            ('steps',      'steps'),
            ('cfg',        'cfg'),
            ('sampler',    'sampler'),
            ('scheduler',  'scheduler'),
            ('positive',   'positive'),
            ('negative',   'negative'),
            ('isBase',     'is_base'),
            ('stepIndex',  'step_index'),
            ('nodeId',     'node_id'),
            ('nodeType',   'node_type'),
        ]
        for i, (js_step, py_step) in enumerate(zip(js_steps, py_steps)):
            for js_key, py_key in field_map:
                if js_key not in js_step:
                    continue
                assert py_step.get(py_key) == js_step[js_key], \
                    f"step[{i}].{py_key}: Python={py_step.get(py_key)!r}, JS={js_step[js_key]!r}"

    def test_annotation_format(self, case):
        annotation = generate_annotation(_get_meta(case))
        assert annotation.startswith('[Generation Info]')
        assert '[Base Sampler -' in annotation
        assert 'Checkpoint:' in annotation

    def test_tags_include_checkpoint(self, case):
        meta = _get_meta(case)
        tags = generate_tags(meta)
        ckpt_no_ext = os.path.splitext(meta.get('checkpoint', ''))[0].lower()
        assert ckpt_no_ext in tags

    def test_tags_exact_match(self, case):
        """Tags must exactly match JS expected (as a set, order-independent)."""
        js = load_js_expected(case['name'])
        if not js or 'tags' not in js:
            pytest.skip('tags not in js expected — run scripts/generate_bridge_expected.js')
        meta = _get_meta(case)
        py_tags = set(generate_tags(meta))
        js_tags = set(js['tags'])
        assert py_tags == js_tags, \
            f"tags mismatch:\n  only in Python: {py_tags - js_tags}\n  only in JS: {js_tags - py_tags}"

    def test_annotation_exact_match(self, case):
        """Annotation must exactly match JS expected."""
        js = load_js_expected(case['name'])
        if not js or 'annotation' not in js:
            pytest.skip('annotation not in js expected — run scripts/generate_bridge_expected.js')
        meta = _get_meta(case)
        py_ann = generate_annotation(meta)
        js_ann = js['annotation']
        assert py_ann == js_ann, \
            f"annotation mismatch:\nPython:\n{py_ann}\n\nJS:\n{js_ann}"
