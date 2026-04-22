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


def read_png_chunks(path):
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

    # Copy bridge-*.png → bridge-*.json (extract prompt + eagle_bridge chunks)
    for fname in os.listdir(cat_fixtures):
        if not (fname.startswith('bridge-') and fname.endswith('.png')):
            continue
        name = fname[:-4]  # strip .png
        src_png = os.path.join(cat_fixtures, fname)
        dst_json = os.path.join(dst_fixtures, f'{name}.json')
        try:
            chunks = read_png_chunks(src_png)
            if 'prompt' not in chunks or 'eagle_bridge' not in chunks:
                print(f'  skip {fname}: missing prompt or eagle_bridge chunk')
                continue
            payload = {
                'prompt': json.loads(chunks['prompt']),
                'eagle_bridge': json.loads(chunks['eagle_bridge']),
            }
            with open(dst_json, 'w', encoding='utf-8') as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            print(f'  fixture: {fname} → tests/fixtures/{name}.json')
            copied_fixtures += 1
        except Exception as e:
            print(f'  error processing {fname}: {e}')

    # Copy bridge-*.json expected files
    for fname in os.listdir(cat_expected):
        if not (fname.startswith('bridge-') and fname.endswith('.json')):
            continue
        src = os.path.join(cat_expected, fname)
        dst = os.path.join(dst_expected, fname)
        shutil.copy2(src, dst)
        print(f'  expected: {fname}')
        copied_expected += 1

    print(f'\nDone: {copied_fixtures} fixtures, {copied_expected} expected files copied.')
    if copied_fixtures == 0:
        print('No PNG fixtures found. Generate them in ComfyUI first.')
        sys.exit(1)


if __name__ == '__main__':
    args = parse_args()
    cat_path = os.path.abspath(args.cat_path)
    if not os.path.isdir(cat_path):
        print(f'Error: comfyui-auto-tagger not found at {cat_path}')
        sys.exit(1)
    print(f'Syncing from: {cat_path}')
    sync(cat_path)
