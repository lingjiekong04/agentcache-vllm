# SPDX-License-Identifier: Apache-2.0
"""Explicitly refresh only the public packaging manifest, never historical pins."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IGNORED = {'.git', 'build', 'reports', '__pycache__', '.pytest_cache'}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--write', action='store_true', required=True)
    p.parse_args()
    mapping = json.loads((ROOT / 'SOURCE_MAP.json').read_text())
    for entry in mapping['files']:
        if hashlib.sha256((ROOT / entry['candidate']).read_bytes()).hexdigest() != entry['sha256']:
            raise ValueError('Original source bytes changed; cannot silently repin')
    files = {}
    for path in sorted(ROOT.rglob('*')):
        rel = path.relative_to(ROOT)
        if not path.is_file() or set(rel.parts) & IGNORED or rel.as_posix() == 'PUBLIC_MANIFEST.json':
            continue
        if path.is_symlink():
            raise ValueError('Symlink not allowed in public candidate')
        files[rel.as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    data = {'scope': 'Public packaging identity, not a new historical GPU acceptance',
            'performance_conclusions_allowed': False, 'files': files}
    (ROOT / 'PUBLIC_MANIFEST.json').write_text(json.dumps(data, indent=2) + '\n', encoding='utf-8')
    print('public_manifest_updated files=' + str(len(files)) + '; rerun check_release.py')


if __name__ == '__main__':
    main()
