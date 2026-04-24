"""
sync_fixtures.py

Copy bridge fixture and expected files from comfyui-auto-tagger to this repo.
Run once after generating new fixtures in comfyui-auto-tagger.

Usage:
    python scripts/sync_fixtures.py [--cat-path PATH]

    --cat-path  Path to comfyui-auto-tagger repo (default: ../comfyui-auto-tagger)
"""
import argparse
import json
import os
import re
import shutil
import struct
import sys


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        '--cat-path',
        default=os.path.join(os.path.dirname(__file__), '..', '..', 'comfyui-auto-tagger'),
        help='Path to comfyui-auto-tagger repository',
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Format-specific metadata readers
# ---------------------------------------------------------------------------

def read_png_chunks(path):
    """Return {'prompt': str, 'eagle_bridge': str, ...} from PNG tEXt chunks."""
    chunks = {}
    with open(path, 'rb') as f:
        f.read(8)
        while True:
            lb = f.read(4)
            if len(lb) < 4:
                break
            length = struct.unpack('>I', lb)[0]
            ct = f.read(4).decode('ascii', errors='replace')
            data = f.read(length)
            f.read(4)
            if ct == 'tEXt':
                ni = data.index(b'\x00')
                key = data[:ni].decode('utf-8', errors='replace')
                val = data[ni + 1:].decode('utf-8', errors='replace')
                chunks[key] = val
    return chunks


def _parse_kv_metadata(text):
    """Parse 'key: {json}' entries from text (EXIF/XMP). Returns dict."""
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


def _read_webp_riff_chunk(path, chunk_types=(b'EXIF', b'XMP ')):
    """Return raw bytes of the first matching RIFF chunk, or None."""
    with open(path, 'rb') as f:
        data = f.read()
    if data[:4] != b'RIFF' or data[8:12] != b'WEBP':
        return None
    offset = 12
    while offset < len(data) - 8:
        chunk_type = data[offset:offset + 4]
        chunk_size = struct.unpack_from('<I', data, offset + 4)[0]
        if chunk_type in chunk_types:
            return data[offset + 8:offset + 8 + chunk_size]
        offset += 8 + chunk_size + (chunk_size % 2)
    return None


def read_webp_metadata(path):
    """
    Extract prompt and eagle_bridge from a WebP file.
    Returns {'prompt': dict, 'eagle_bridge': dict} or None.
    """
    raw = _read_webp_riff_chunk(path)
    if raw is None:
        return None
    text = raw.decode('latin-1', errors='replace')
    meta = _parse_kv_metadata(text)
    if 'prompt' not in meta or 'eagle_bridge' not in meta:
        return None
    return {'prompt': meta['prompt'], 'eagle_bridge': meta['eagle_bridge']}


def read_jpeg_metadata(path):
    """
    Extract prompt and eagle_bridge from a JPEG file via EXIF tags.
    Tag mapping written by executor.py _build_jpeg_exif:
      0x010F (Make=271)   → "Prompt: {json}"  or "workflow: {json}"
      0x013B (Artist=315) → "eagle_bridge: {json}"
    Returns {'prompt': dict, 'eagle_bridge': dict} or None.
    """
    try:
        from PIL import Image
    except ImportError:
        print('  warning: Pillow not installed; skipping JPEG metadata read')
        return None

    try:
        img = Image.open(path)
        exif_data = img.getexif()
    except Exception:
        return None

    MAKE_TAG = 0x010F    # 271
    ARTIST_TAG = 0x013B  # 315

    make_val = exif_data.get(MAKE_TAG, '')
    artist_val = exif_data.get(ARTIST_TAG, '')

    combined = f"{make_val}\n{artist_val}"
    meta = _parse_kv_metadata(combined)
    if 'prompt' not in meta or 'eagle_bridge' not in meta:
        return None
    return {'prompt': meta['prompt'], 'eagle_bridge': meta['eagle_bridge']}


# Map extension → reader function
_READERS = {
    '.png': lambda path: _png_to_payload(path),
    '.webp': read_webp_metadata,
    '.jpg': read_jpeg_metadata,
    '.jpeg': read_jpeg_metadata,
}


def _png_to_payload(path):
    chunks = read_png_chunks(path)
    if 'prompt' not in chunks or 'eagle_bridge' not in chunks:
        return None
    return {
        'prompt': json.loads(chunks['prompt']),
        'eagle_bridge': json.loads(chunks['eagle_bridge']),
    }


# ---------------------------------------------------------------------------
# Sync
# ---------------------------------------------------------------------------

def sync(cat_path):
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cat_fixtures = os.path.join(cat_path, 'tests', 'fixtures')
    cat_expected = os.path.join(cat_path, 'tests', 'expected')
    dst_fixtures = os.path.join(repo_root, 'tests', 'fixtures')
    dst_expected = os.path.join(repo_root, 'tests', 'expected')

    os.makedirs(dst_fixtures, exist_ok=True)
    os.makedirs(dst_expected, exist_ok=True)

    copied_fixtures = 0
    copied_expected = 0

    for fname in sorted(os.listdir(cat_fixtures)):
        if not fname.startswith('bridge-'):
            continue
        name, ext = os.path.splitext(fname)
        reader = _READERS.get(ext.lower())
        if reader is None:
            continue

        src = os.path.join(cat_fixtures, fname)
        dst_json = os.path.join(dst_fixtures, f'{name}.json')
        try:
            payload = reader(src)
            if payload is None:
                print(f'  skip {fname}: missing prompt or eagle_bridge metadata')
                continue
            with open(dst_json, 'w', encoding='utf-8') as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            print(f'  fixture: {fname} → tests/fixtures/{name}.json')
            copied_fixtures += 1
        except Exception as e:
            print(f'  error processing {fname}: {e}')

    # Copy bridge-*.json expected files
    for fname in sorted(os.listdir(cat_expected)):
        if not (fname.startswith('bridge-') and fname.endswith('.json')):
            continue
        src = os.path.join(cat_expected, fname)
        dst = os.path.join(dst_expected, fname)
        shutil.copy2(src, dst)
        print(f'  expected: {fname}')
        copied_expected += 1

    print(f'\nDone: {copied_fixtures} fixtures, {copied_expected} expected files copied.')
    if copied_fixtures == 0:
        print('No fixtures found. Generate them in ComfyUI first.')
        sys.exit(1)


if __name__ == '__main__':
    args = parse_args()
    cat_path = os.path.abspath(args.cat_path)
    if not os.path.isdir(cat_path):
        print(f'Error: comfyui-auto-tagger not found at {cat_path}')
        sys.exit(1)
    print(f'Syncing from: {cat_path}')
    sync(cat_path)
