"""Strata-Qwen36: the fork's folders stay apart from an upstream Strata install on the same PC, and its model table is
consistent.

    python -m unittest discover -s tools -p test_setup_qwen36.py
"""
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import setup as S  # noqa: E402
import setup_qwen36 as Q  # noqa: E402


class Folders(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.old = {k: os.environ.get(k) for k in ("APPDATA", "XDG_CONFIG_HOME")}
        os.environ["APPDATA"] = os.environ["XDG_CONFIG_HOME"] = self.tmp.name

    def tearDown(self):
        for k, v in self.old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        self.tmp.cleanup()

    def test_settings_folder_is_the_forks(self):
        p = S.settings_path()
        self.assertEqual(p.parent.name.lower(), "strata-qwen36")
        self.assertNotEqual(p.parent.name.lower(), "strata")

    def test_upstream_installs_are_not_adopted(self):
        base = Path(self.tmp.name)
        upstream = base / "Strata"                     # upstream: setup.py only
        fork = base / "Strata-Qwen36-copy"             # another copy of this fork
        for d in (upstream, fork):
            d.mkdir()
            (d / "setup.py").write_text("")
        (fork / "setup_qwen36.py").write_text("")
        found = S.other_installs({"installs": [str(upstream), str(fork)]})
        self.assertIn(fork.resolve(), found)
        self.assertNotIn(upstream.resolve(), found)

    def test_data_folder_name(self):
        self.assertEqual(S.DATA_DIR, "Strata-Qwen36-data")
        self.assertNotEqual(S.DATA_DIR, "Strata-data")


class Models(unittest.TestCase):
    def test_table(self):
        for name, d in Q.MODELS.items():
            self.assertTrue(d["file"].endswith(".gguf"), name)
            self.assertEqual(len(d["sha256"]), 64, name)
            self.assertGreater(d["bytes"], 1e9, name)

    def test_port_differs_from_upstream(self):
        self.assertNotEqual(Q.DEFAULT_PORT, 8080)

    def test_contexts_within_training(self):
        self.assertTrue(all(c <= 262144 for c in Q.CONTEXTS))


class EffortBudgets(unittest.TestCase):
    """serve/server.py's effort_budget: Qwen3.6's template has no reasoning levels, so the config maps them to thinking
    budgets; without the config nothing changes (upstream's models keep their template's levels)."""

    @classmethod
    def setUpClass(cls):
        from serve.frontend import ChatTemplate
        from serve.server import ByteTokenizer, MockEngine, Service
        tok = ByteTokenizer()
        cls.svc = Service(MockEngine(tok, "ok"), tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))

    def tearDown(self):
        self.svc.effort_budgets = {}
        self.svc.reasoning_budget_tokens = 0

    def test_off_without_config(self):
        self.assertIsNone(self.svc.reasoning_budget({"reasoning_effort": "low"}))

    def test_levels(self):
        self.svc.effort_budgets = dict(Q.EFFORT_BUDGETS)
        self.assertEqual(self.svc.reasoning_budget({"reasoning_effort": "low"}), 1024)
        self.assertEqual(self.svc.reasoning_budget({"reasoning": {"effort": "Medium"}}), 4096)
        self.assertIsNone(self.svc.reasoning_budget({"reasoning_effort": "high"}))
        self.assertEqual(self.svc.reasoning_budget({"reasoning_effort": "high",
                                                    "chat_template_kwargs": {"reasoning_effort": "low"}}), 1024)
        self.assertEqual(self.svc.reasoning_budget({"output_config": {"effort": "low"}}), 1024)
        self.assertEqual(self.svc.reasoning_budget({"thinking": {"type": "enabled", "budget_tokens": 3000}}), 3000)

    def test_explicit_budget_wins(self):
        self.svc.effort_budgets = dict(Q.EFFORT_BUDGETS)
        self.svc.reasoning_budget_tokens = 500
        self.assertEqual(self.svc.reasoning_budget({"reasoning_effort": "low", "reasoning_budget_tokens": 77}), 77)
        self.assertEqual(self.svc.reasoning_budget({"reasoning_effort": "low"}), 1024)
        self.assertEqual(self.svc.reasoning_budget({}), 500)


if __name__ == "__main__":
    unittest.main()
