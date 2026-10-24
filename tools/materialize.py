# SPDX-License-Identifier: Apache-2.0
"""Reconstruct the pinned upstream baseline plus the reviewed source overlay."""
import argparse
import hashlib
import json
import os
import tarfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]


def digest(data):
    return hashlib.sha256(data).hexdigest()


def filesystem_path(path):
    # The upstream tree has filenames exceeding Win32 MAX_PATH in long workspaces.
    if os.name == 'nt' and not str(path).startswith('\\\\?\\'):
        return Path('\\\\?\\' + str(path.absolute()))
    return path


def safe_name(name):
    p = PurePosixPath(name)
    if p.is_absolute() or '..' in p.parts or '\\' in name or ':' in name or p.as_posix() != name.rstrip('/'):
        raise ValueError('Unsafe archive path: ' + name)
    return p


def baseline_files(archive):
    config = json.loads((ROOT / 'upstream/baseline.json').read_text(encoding='utf-8'))
    if digest(archive.read_bytes()) != config['sha256']:
        raise ValueError('Baseline SHA256 mismatch; do not change the pin')
    files, total = {}, 0
    with tarfile.open(archive, 'r:gz') as tar:
        for member in tar.getmembers():
            p = safe_name(member.name)
            if not p.parts or p.parts[0] != config['archive_root'] or not (member.isfile() or member.isdir()):
                raise ValueError('Unexpected archive root or member type')
            if member.isdir():
                continue
            name = '/'.join(p.parts[1:])
            total += member.size
            if not name or name in files or member.size > 32 * 1024**2 or total > 512 * 1024**2:
                raise ValueError('Duplicate or oversized archive entry')
            files[name] = tar.extractfile(member).read()
    return files


def verify_tree(output):
    output = filesystem_path(output)
    record = json.loads((output / 'materialization.json').read_text(encoding='utf-8'))
    if record['overlay_sha256'] != digest((ROOT / 'SOURCE_MAP.json').read_bytes()):
        raise ValueError('Materialized tree belongs to a different overlay')
    for name, expected in record['files'].items():
        if digest((output / name).read_bytes()) != expected:
            raise ValueError('Materialized source changed: ' + name)
    return record


def materialize(archive, output):
    output = filesystem_path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError('Output exists; preserve it and choose a new directory')
    files = baseline_files(archive)
    mapping = json.loads((ROOT / 'SOURCE_MAP.json').read_text(encoding='utf-8'))
    for entry in mapping['files']:
        path = ROOT / entry['candidate']
        data = path.read_bytes()
        if digest(data) != entry['sha256']:
            raise ValueError('Overlay changed: ' + entry['candidate'])
        files[entry['candidate'].removeprefix('overlay/')] = data
    # This is a NEW packaging manifest, not the historical accepted manifest.
    source_manifest = {'baseline_tag': 'v0.8.3',
        'baseline_archive_sha256': json.loads((ROOT / 'upstream/baseline.json').read_text())['sha256'],
        'scope': 'Public materialization; unchanged accepted core, not a new GPU acceptance',
        'files': {entry['candidate'].removeprefix('overlay/'): entry['sha256'] for entry in mapping['files']}}
    files['agentcache/source_manifest.json'] = (json.dumps(source_manifest, indent=2) + '\n').encode()
    output.mkdir(parents=True, exist_ok=False)
    for name, data in files.items():
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open('xb') as stream:
            stream.write(data)
    record = {'baseline_sha256': source_manifest['baseline_archive_sha256'],
        'overlay_sha256': digest((ROOT / 'SOURCE_MAP.json').read_bytes()),
        'files': {name: digest(data) for name, data in files.items()},
        'gpu_validated': False, 'performance_conclusions_allowed': False}
    with (output / 'materialization.json').open('x', encoding='utf-8') as stream:
        json.dump(record, stream, indent=2)
    verify_tree(output)
    return record


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--archive', type=Path, required=True)
    p.add_argument('--output', type=Path, default=ROOT / 'build/vllm-0.8.3')
    args = p.parse_args()
    result = materialize(args.archive.resolve(), args.output.absolute())
    print('materialized=true files=' + str(len(result['files'])) + ' no_gpu=true')
