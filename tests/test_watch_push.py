"""deploy/watch-push (vibehash.org/watch's alerts Worker): its own tests, in Node, against a real SQLite database.
Skipped where there's no Node 22.5 or later (node:sqlite)."""
import shutil
import subprocess
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent / "deploy" / "watch-push"


def node_with_sqlite() -> str | None:
    node = shutil.which("node")
    if not node:
        return None
    probe = subprocess.run([node, "-e", "require('node:sqlite')"], capture_output=True, timeout=30)
    return node if probe.returncode == 0 else None


class WatchPushTests(unittest.TestCase):
    @unittest.skipUnless(node_with_sqlite(), "needs Node 22.5+ (node:sqlite)")
    def test_the_worker(self):
        out = subprocess.run([node_with_sqlite(), "test.mjs"], cwd=HERE, capture_output=True, text=True, timeout=120)
        self.assertEqual(out.returncode, 0, out.stdout + out.stderr)
        self.assertIn(" passed", out.stdout)

    def test_the_page_and_the_worker_agree_on_the_address(self):
        page = (HERE.parent.parent / "docs" / "site" / "watch.html").read_text(encoding="utf-8")
        sw = (HERE.parent.parent / "docs" / "site" / "watch-sw.js").read_text(encoding="utf-8")
        config = (HERE / "wrangler.jsonc").read_text(encoding="utf-8")
        self.assertIn('"pattern": "watch-api.vibehash.org"', config)
        for text in (page, sw):
            self.assertIn('"https://watch-api.vibehash.org"', text)
