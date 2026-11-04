"""
Ricursive (Ripple) integration smoke test on a tiny random checkpoint: the arch resolves, the
PeriNorm/SWA/NoPE graph loads, and the value-residual seed reaches the V projections. Full
numerical parity against the reference modeling_ricursive.py was checked separately on the real
checkpoint; this keeps the architecture and core value-residual plumbing from regressing.
"""

import torch

from exllamav3 import Config, Model
from testlib.tiny_models import make_ricursive_checkpoint


@torch.inference_mode()
def test_ricursive_forward(tmp_path, device):
    model_dir = make_ricursive_checkpoint(tmp_path / "ricursive")
    cfg = Config.from_directory(model_dir)
    cfg.override_dynamic_seq_len(64)
    model = Model.from_config(cfg)
    model.load(torch.device(device))
    try:
        ids = torch.randint(0, 512, (1, 64))
        params = {"attn_mode": "flash_attn_nc"}
        logits = model.forward(ids, params)[0]
        assert logits.shape[-1] == 512
        assert torch.isfinite(logits).all()

        # The embedding stashed the scaled layer-0 rows for value residuals
        assert params["valres_seed"].shape == (1, 64, 128)

        local = model.find_module("model.layers.0.self_attn")
        glob = model.find_module("model.layers.2.self_attn")
        assert local.valres_key == "value_res_lambda"
        assert local.valres_blend is not None
        assert local.rope is not None
        assert glob.rope is None
    finally:
        model.unload()
