# SPDX-License-Identifier: Apache-2.0
import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('materialize', ROOT / 'tools/materialize.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


class Contracts(unittest.TestCase):
    def test_archive_paths(self):
        for name in ('../a', '/a', 'a/../b', 'C:/a', 'a\\b'):
            with self.assertRaises(ValueError):
                m.safe_name(name)

    def test_existing_output_never_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(FileExistsError):
                m.materialize(Path('not-needed.tar.gz'), Path(directory))

    def test_source_map_hashes(self):
        for entry in json.loads((ROOT / 'SOURCE_MAP.json').read_text())['files']:
            self.assertEqual(hashlib.sha256((ROOT / entry['candidate']).read_bytes()).hexdigest(), entry['sha256'])

    def test_evidence_counts_and_limits(self):
        r = json.loads((ROOT / 'evidence/historical-verification.json').read_text())
        self.assertEqual(r['serial_requests_verified'], 668)
        self.assertEqual(r['paired_serial_requests'], 334)
        self.assertEqual(r['stream_disconnects_verified'], 4)
        self.assertFalse(r['performance_conclusions_allowed'])
        for run in r['lifecycles'].values():
            self.assertEqual(run['concurrent_completed'], 16)
            self.assertEqual(run['pressure_completed'], 4)
            self.assertEqual(run['preemptions_observed'], 2)

    def test_historical_manifest_is_distinct(self):
        data = (ROOT / 'evidence/accepted-source-manifest.json').read_bytes()
        self.assertEqual(hashlib.sha256(data).hexdigest(), 'c68827f0cfaf410f2e5b2fc66b5844f4fc73253cbb49e1373da955b180c55531')


if __name__ == '__main__':
    unittest.main(verbosity=2)
