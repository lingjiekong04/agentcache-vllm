# SPDX-License-Identifier: Apache-2.0
"""Verify public file hashes, source provenance, links and basic privacy rules."""
import ast
import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IGNORED = {'.git', 'build', 'reports', '__pycache__', '.pytest_cache'}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    manifest = json.loads((ROOT / 'PUBLIC_MANIFEST.json').read_text(encoding='utf-8'))
    current = {p.relative_to(ROOT).as_posix() for p in ROOT.rglob('*') if p.is_file()
               and not set(p.relative_to(ROOT).parts) & IGNORED and p.name != 'PUBLIC_MANIFEST.json'}
    if current != set(manifest['files']):
        raise ValueError('Public file list changed; regenerate manifest in a NEW release')
    for name, expected in manifest['files'].items():
        path = ROOT / name
        if path.is_symlink() or sha(path) != expected:
            raise ValueError('Public file hash differs: ' + name)
        if path.suffix == '.py':
            ast.parse(path.read_bytes(), filename=name, feature_version=(3, 11))
        if path.suffix in ('.py', '.md', '.json', '.csv', '.sh', '.yml', '.ps1'):
            text = path.read_text(encoding='utf-8')
            blocked = '|'.join([r'GPU-[0-9a-f]{8}-[0-9a-f-]+', '/' + 'konglingjie' + '/',
                r'C:\\Users\\', r'-----BEGIN (?:RSA |OPENSSH )?PRIVATE KEY-----',
                r'gh[pousr]_[A-Za-z0-9]{20,}', r'hf_[A-Za-z0-9]{20,}'])
            if re.search(blocked, text):
                raise ValueError('Sensitive literal pattern found: ' + name)
            if path.suffix == '.md':
                for target in re.findall(r'\]\(([^)]+)\)', text):
                    if target.startswith(('http:', 'https:', '#', 'mailto:')):
                        continue
                    target = target.split('#', 1)[0].strip('<>')
                    if target and not (path.parent / target).exists():
                        raise ValueError('Broken local link: ' + name + ' -> ' + target)
    accepted = json.loads((ROOT / 'evidence/accepted-source-manifest.json').read_text())['files']
    mapping = json.loads((ROOT / 'SOURCE_MAP.json').read_text())['files']
    for entry in mapping:
        if sha(ROOT / entry['candidate']) != entry['sha256']:
            raise ValueError('Source provenance mismatch')
        name = entry['candidate'].removeprefix('overlay/')
        if name in accepted and accepted[name] != entry['sha256']:
            raise ValueError('Accepted core changed')
    results = json.loads((ROOT / 'evidence/results.json').read_text())
    assert results['performance_conclusions_allowed'] is False
    for comparison in results['comparisons'].values():
        for phases in comparison['summary'].values():
            for phase in phases.values():
                q, h = phase['cache_queries'], phase['cache_hits']
                assert q > 0 and 0 <= h <= q and abs(h / q - phase['hit_ratio']) < 1e-12
    print(json.dumps({'public_integrity_passed': True, 'files': len(current),
        'source_bytes_verified': len(mapping), 'scope': 'Integrity and limited screening, not a complete security audit',
        'performance_conclusions_allowed': False}))


if __name__ == '__main__':
    main()
