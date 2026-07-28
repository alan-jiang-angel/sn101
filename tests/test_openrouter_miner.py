import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]


def load_openrouter_miner_module():
    pkg = types.ModuleType("tag101")
    pkg.__path__ = [str(ROOT)]
    sys.modules["tag101"] = pkg

    tasks_pkg = types.ModuleType("tag101.tasks")
    tasks_pkg.__path__ = [str(ROOT / "tasks")]
    sys.modules["tag101.tasks"] = tasks_pkg

    framework_pkg = types.ModuleType("tag101.tasks.framework")
    framework_pkg.__path__ = [str(ROOT / "tasks" / "framework")]
    sys.modules["tag101.tasks.framework"] = framework_pkg

    base_module = types.ModuleType("tag101.tasks.framework.base")
    class TaskHandler:  # pragma: no cover - stub only for import
        pass
    base_module.TaskHandler = TaskHandler
    sys.modules["tag101.tasks.framework.base"] = base_module

    sn101_module = types.ModuleType("tag101.tasks.sn101")
    sn101_module.KIND = "sn101.tags.v1"
    sn101_module.SPEC_VERSION = "v1"
    sn101_module.score_answers = lambda *args, **kwargs: None
    sys.modules["tag101.tasks.sn101"] = sn101_module

    preprocessing_module = types.ModuleType(
        "tag101.tasks.sn101_reference.core.scoring.preprocessing"
    )

    def normalize_tag(tag: str) -> str:
        return " ".join(str(tag).strip().lower().split())

    preprocessing_module.normalize_tag = normalize_tag
    sys.modules["tag101.tasks.sn101_reference.core.scoring.preprocessing"] = preprocessing_module

    protocol_module = types.ModuleType("tag101.protocol")
    class TaskEnvelope:  # pragma: no cover - stub only for import
        pass
    protocol_module.TaskEnvelope = TaskEnvelope
    sys.modules["tag101.protocol"] = protocol_module

    numpy_stub = types.ModuleType("numpy")
    class _Array(list):
        pass

    def array(values, dtype=None):
        return _Array(values)

    def ndim(value):
        return 1 if isinstance(value, list) and value and isinstance(value[0], (list, tuple)) else 0

    numpy_stub.array = array
    numpy_stub.ndim = ndim
    numpy_stub.dot = lambda a, b: 0.0
    sys.modules["numpy"] = numpy_stub

    sentence_transformers_stub = types.ModuleType("sentence_transformers")
    sentence_transformers_stub.SentenceTransformer = object
    sys.modules["sentence_transformers"] = sentence_transformers_stub

    spec = importlib.util.spec_from_file_location(
        "tag101.tasks.openrouter_miner",
        ROOT / "tasks" / "openrouter_miner.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["tag101.tasks.openrouter_miner"] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


OpenRouterMiner = load_openrouter_miner_module().OpenRouterMiner


class _DummyModel:
    def encode(self, texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False):
        import numpy as np

        vectors = []
        for text in texts:
            lowered = text.lower()
            if lowered == "bitcoin":
                vectors.append([1.0, 0.0, 0.0])
            elif lowered == "crypto":
                vectors.append([0.9, 0.1, 0.0])
            elif lowered == "investment":
                vectors.append([0.2, 0.9, 0.0])
            else:
                vectors.append([0.0, 0.0, 1.0])
        return np.array(vectors, dtype=float)


class OpenRouterMinerTests(unittest.TestCase):
    def test_select_top_tags_prefers_relevant_and_diverse_tags(self):
        miner = OpenRouterMiner(n_tags=3)

        with patch("sentence_transformers.SentenceTransformer", return_value=_DummyModel()):
            selected = miner._select_top_tags(
                "bitcoin and crypto investing",
                ["investment", "bitcoin", "crypto", "finance"],
            )

        self.assertEqual(selected[0], "bitcoin")
        self.assertTrue(set(selected[1:]).issubset({"crypto", "investment"}))
        self.assertEqual(len(selected), 3)


if __name__ == "__main__":
    unittest.main()
