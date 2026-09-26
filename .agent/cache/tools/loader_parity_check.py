"""DS-02 parity check: the legacy AttentionModel rebuilt by utils/model/loader.py must
reproduce the training AttentionModelPolicy's encoder output for the same weights.

usage (repo root): .venv/bin/python .agent/cache/tools/loader_parity_check.py <checkpoint_dir>
"""
import os
import sys

sys.path.insert(0, os.getcwd())
import torch
from omegaconf import OmegaConf

from logic.src.pipeline.features.train.model_factory.builder import _init_policy
from logic.src.utils.model.loader import load_model

ckpt_dir = sys.argv[1]
legacy, _ = load_model(ckpt_dir)
cfg = OmegaConf.load(os.path.join(ckpt_dir, "config.yaml"))
policy = _init_policy(cfg, None)
state = torch.load(os.path.join(ckpt_dir, "epoch-1.pt"), map_location="cpu", weights_only=False)["state_dict"]
policy.load_state_dict({k[len("policy."):]: v for k, v in state.items() if k.startswith("policy.")}, strict=True)
legacy.eval(); policy.eval()

enc_l, enc_p = legacy.encoder, policy.encoder
print("norm (legacy / policy):", type(enc_l.layers[0].norm1.normalizer).__name__, "/", type(enc_p.layers[0].norm1.normalizer).__name__)
torch.manual_seed(0)
h = torch.randn(3, 21, enc_p.layers[0].norm1.normalizer.normalized_shape[0] if hasattr(enc_p.layers[0].norm1.normalizer, "normalized_shape") else 32)
with torch.no_grad():
    out_l, out_p = enc_l(h), enc_p(h)
out_l = out_l[0] if isinstance(out_l, tuple) else out_l
out_p = out_p[0] if isinstance(out_p, tuple) else out_p
diff = (out_l - out_p).abs().max().item()
print(f"max |encoder(legacy) - encoder(policy)| = {diff:.3e}")
assert diff < 1e-5, "legacy model does not reproduce the training policy"
print("PARITY OK")
