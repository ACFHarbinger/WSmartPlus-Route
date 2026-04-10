---
description: Add a new neural network architecture (encoder, decoder, or full model) to logic/src/models/.
---

You are an AI Research Scientist adding a new neural architecture to WSmart+ Route's NCO framework.

## Directory Map

```
logic/src/models/
├── modules/      # Atomic components (attention, normalization, graph layers)
├── subnets/      # Encoders, decoders, predictors
├── policies/     # Policy wrappers (HGS-based, local search)
└── core/         # Base classes and registry
```

## Implementation Steps

### 1. Read Existing Reference
- For attention-based models → read the AM or TAM model in `logic/src/models/`.
- For graph-based → read GNN layers in `logic/src/models/modules/`.
- For encoder-decoder pattern → read existing encoder/decoder in `logic/src/models/subnets/`.

### 2. Normalization (CRITICAL)
Use the project normalization module — NEVER use `nn.LayerNorm` directly:
```python
from logic.src.models.modules.normalization import Normalization
self.norm = Normalization(embed_dim, normalization='batch')  # or 'instance', 'layer'
```

### 3. Masking (CRITICAL for decoders)
Apply masking before softmax in any node selection:
```python
from logic.src.utils.functions.boolmask import mask_long2bool, mask_long_scatter

logits = self.compute_logits(state)
logits = logits.masked_fill(invalid_mask, float('-inf'))
probs = torch.softmax(logits, dim=-1)
action = torch.multinomial(probs, 1)
```

### 4. Device Management
```python
from logic.src.utils.configs.setup_utils import get_device
device = get_device(cuda_enabled=True)
# Pass device through; do NOT hardcode .cuda()
```

### 5. Type Hints and Shape Comments
```python
from typing import Optional, Tuple
import torch

def forward(
    self,
    h: torch.Tensor,          # (B, N, embed_dim)
    mask: Optional[torch.Tensor] = None,  # (B, N) bool
) -> Tuple[torch.Tensor, torch.Tensor]:   # logits (B, N), context (B, embed_dim)
    B, N = h.size()[:2]  # B=batch, N=nodes
    ...
```

### 6. Weight Naming
- Learnable weight matrices: prefix `W_` (e.g., `self.W_query`, `self.W_key`, `self.W_val`).
- Flattened tensors: suffix `_flat` (e.g., `h_flat`).

## Testing

Add tests in `logic/test/unit/models/test_<model_name>.py`:
- Test forward pass with dummy input tensors.
- Test output shapes are correct for various batch and graph sizes.
- Test that masked nodes get zero probability.
- Run: `python main.py test_suite --module test_models`

## Performance Notes
- For large graphs (N > 100): use `logic/src/models/modules/efficient_graph_convolution.py`.
- Target GPUs: NVIDIA RTX 3090 Ti (24 GB) and RTX 4080 laptop (12 GB).
- Batch sizes: 256 for 12 GB VRAM, 512–1024 for 24 GB VRAM.
