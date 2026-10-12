import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / ".vendor"))


def test_neural_gate_does_not_leave_torch_on_one_thread():
    pytest.importorskip("silero_vad")
    import torch

    from trcs_eval import gate

    before = torch.get_num_threads()
    gate.neural_gate(np.zeros(16000, dtype=np.float32))
    assert torch.get_num_threads() == before
