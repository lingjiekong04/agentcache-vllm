# SPDX-License-Identifier: Apache-2.0
"""Cost-bounded stdlib/extracted checks. Does not run a GPU or native serving."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
import materialize

ROOT = Path(__file__).resolve().parents[1]


def run(archive, output):
    output.mkdir(parents=True, exist_ok=False)
    status = {'passed': False, 'scope': 'Standard-library and extracted metadata only',
              'native_pytest_rerun': False, 'gpu_rerun': False,
              'performance_conclusions_allowed': False, 'checks': []}
    try:
        source = ROOT / 'build/vllm-0.8.3'
        if source.exists():
            materialize.verify_tree(source)
        else:
            materialize.materialize(archive, source)
        commands = [
            ('release-tools', [str(ROOT / 'tests/test_release_tools.py')]),
            ('tool-contracts', [str(source / 'agentcache/test_tools.py')]),
            ('config-contracts', [str(source / 'agentcache/test_config_fix.py')]),
            ('spawn-mocks', [str(source / 'agentcache/test_spawn_v2.py')]),
            ('cpu-extracted', [str(source / 'agentcache/check_cpu.py'), '--extracted',
                '--archive', str(archive), '--reference', str(source / 'agentcache/reference'),
                '--output', str(output / 'cpu-extracted.json')])]
        temp = output / 'tmp'
        temp.mkdir()
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONIOENCODING='utf-8', CUDA_VISIBLE_DEVICES='',
                   TMPDIR=str(temp), TEMP=str(temp), TMP=str(temp))
        for name, args in commands:
            command = [sys.executable, '-S', '-B', *args]
            p = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True, encoding='utf-8', errors='replace')
            with (output / (name + '.log')).open('x', encoding='utf-8') as stream:
                stream.write(p.stdout + p.stderr)
            status['checks'].append({'name': name, 'exit_code': p.returncode})
            print(name + ' exit=' + str(p.returncode), flush=True)
            if p.returncode:
                raise RuntimeError('Check failed; preserve evidence: ' + name)
        status['cpu'] = json.loads((output / 'cpu-extracted.json').read_text())
        status['passed'] = True
    except BaseException as error:
        status['error'] = type(error).__name__ + ': ' + str(error)
        raise
    finally:
        with (output / 'result.json').open('x', encoding='utf-8') as stream:
            json.dump(status, stream, indent=2)
    return status


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--archive', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    run(args.archive.resolve(), args.output.absolute())
