"""
test_bridge_fixtures.py

Fixture-based tests for metadata extraction (extract_metadata) against real
ComfyUI images. Python output is compared directly against JS parser output
(source of truth) by running node scripts/parse-image-json.js at test time.

Cross-repo contract
-------------------
Fixture images live in comfyui-auto-tagger/tests/fixtures/.
Set COMFYUI_AUTO_TAGGER_PATH to point to a comfyui-auto-tagger checkout
(defaults to ../../comfyui-auto-tagger).

Tests are skipped automatically when:
  - COMFYUI_AUTO_TAGGER_PATH is unavailable
  - Node.js is not installed

To add a new fixture:
  1. Generate an image via ComfyUI with eagle-metadata-bridge node
  2. Copy PNG/WebP to comfyui-auto-tagger/tests/fixtures/bridge-<name>.{png,webp}
  3. Add a test case to TEST_CASES below
"""
import json
import os
import shutil
import struct
import subprocess
import pytest

try:
    from PIL import Image
    _PIL_AVAILABLE = True
except ImportError:
    _PIL_AVAILABLE = False

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
PARSE_SCRIPT = os.path.join(_cat_path, 'scripts', 'parse-image-json.js')

# ---------------------------------------------------------------------------
# Node.js availability
# ---------------------------------------------------------------------------

_NODE = shutil.which('node')

def _js_parse(image_path):
    """
    Run the JS parser on image_path via Node.js.
    Returns parsed metadata dict, or None if Node is unavailable / parse fails.
    """
    if not _NODE:
        return None
    if not os.path.exists(PARSE_SCRIPT):
        return None
    try:
        result = subprocess.run(
            [_NODE, PARSE_SCRIPT, image_path],
            capture_output=True, text=True, timeout=15
        )
        if result.returncode != 0:
            return None
        return json.loads(result.stdout)
    except Exception:
        return None

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

def _fixture_path(case):
    fmt = case['_fmt']
    return os.path.join(CAT_FIXTURES_DIR, case[f'fixture_{fmt}'])

def _get_fixture(case):
    """Return loaded fixture dict or skip if unavailable."""
    path = _fixture_path(case)
    data = load_fixture_from_image(path)
    if data is None:
        pytest.skip(f'fixture not available: {path} (set COMFYUI_AUTO_TAGGER_PATH)')
    return data

def _get_meta(case):
    fixture = _get_fixture(case)
    return extract_metadata(fixture['prompt'], fixture['eagle_bridge']['final_node_id'])

def _get_js(case):
    """Return JS parser output or skip if Node unavailable."""
    js = _js_parse(_fixture_path(case))
    if js is None:
        pytest.skip('Node.js not available or parse-image-json.js not found')
    return js

# ---------------------------------------------------------------------------
# Parametrized tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize('case', _PARAMETRIZE, ids=[c['_id'] for c in _PARAMETRIZE])
class TestBridgeFixtures:

    def test_fixture_image_exists(self, case):
        path = _fixture_path(case)
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

    def test_core_fields_match_js(self, case):
        """Top-level scalar fields must match JS parser output."""
        js = _get_js(case)
        meta = _get_meta(case)
        for field in ('checkpoint', 'seed', 'steps', 'cfg', 'sampler', 'scheduler',
                      'positive', 'negative'):
            if field not in js:
                continue
            assert meta.get(field) == js[field], \
                f"field '{field}': Python={meta.get(field)!r}, JS={js[field]!r}"
        if 'loras' in js:
            assert set(meta.get('loras', [])) == set(js['loras']), \
                f"loras: Python={meta.get('loras')!r}, JS={js['loras']!r}"

    def test_generation_steps_match_js(self, case):
        """Each generation step must match JS parser output field-by-field."""
        js = _get_js(case)
        if 'generationSteps' not in js:
            pytest.skip('generationSteps not in JS output')
        meta = _get_meta(case)
        py_steps = meta.get('generation_steps', [])
        js_steps = js['generationSteps']
        assert len(py_steps) == len(js_steps), \
            f"step count: Python={len(py_steps)}, JS={len(js_steps)}"
        # camelCase (JS) → snake_case (Python) field map
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

    def test_eagle_bridge_final_node_id_matches_js(self, case):
        """eagle_bridge.final_node_id in fixture must match JS parser output."""
        js = _get_js(case)
        if 'eagle_bridge' not in js:
            pytest.skip('eagle_bridge not in JS output')
        fixture = _get_fixture(case)
        assert fixture['eagle_bridge']['final_node_id'] == js['eagle_bridge']['final_node_id']
