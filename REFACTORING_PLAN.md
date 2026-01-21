# Refactoring Plan: RL4CO-Style Modular Architecture

> **Created**: January 2026
> **Objective**: Refactor `logic/src/` to adopt RL4CO's modular architecture patterns

---

## Overview

This plan transforms WSmart-Route's logic layer from a problem-coupled architecture to a modular, RL4CO-style design using:

- **TensorDict** for unified state management
- **PyTorch Lightning** for training orchestration
- **Hydra** for configuration management
- **Environment abstraction** for problem-agnostic interfaces

---

## Design Decisions

| Decision | Choice |
|----------|--------|
| Training Framework | PyTorch Lightning |
| Migration Strategy | Full backward compatibility with deprecation warnings |
| Configuration System | Hydra with hierarchical configs |

---

## New Directory Structure

```
logic/src/
├── envs/                       # NEW: Environment abstraction
│   ├── __init__.py            # Registry and get_env()
│   ├── base.py                # RL4COEnvBase
│   ├── generators.py          # Data generators
│   ├── vrpp.py               # VRPPEnv
│   └── wcvrp.py              # WCVRPEnv
│
├── models/
│   ├── policies/              # NEW: Policy base classes
│   │   └── base.py           # ConstructivePolicy
│   ├── embeddings/            # NEW: Init embeddings
│   │   └── __init__.py       # Registry
│   ├── modules/              # KEEP: Neural building blocks
│   ├── subnets/              # KEEP: Encoders/decoders
│   └── attention_model.py    # REFACTOR: Inherit from ConstructivePolicy
│
├── configs/                   # NEW: Configuration dataclasses
│   └── __init__.py
│
├── pipeline/
│   ├── rl/                   # REFACTOR: Modular RL
│   │   ├── base.py           # RL4COLitModule
│   │   ├── baselines.py
│   │   └── reinforce.py
│   └── trainer.py            # NEW: WSTrainer
│
├── data/
│   └── datasets.py           # NEW: TensorDict datasets
│
└── tasks/                    # DEPRECATE: Keep for backward compat
```

---

## Implementation Phases

### Phase 1: Environment Abstraction ✅

| File | Status |
|------|--------|
| `logic/src/envs/base.py` | ✅ Created |
| `logic/src/envs/generators.py` | ✅ Created |
| `logic/src/envs/vrpp.py` | ✅ Created |
| `logic/src/envs/wcvrp.py` | ✅ Created |
| `logic/src/envs/__init__.py` | ✅ Created |

### Phase 2: Model Architecture ✅

| File | Status |
|------|--------|
| `logic/src/models/policies/base.py` | ✅ Created |
| `logic/src/models/policies/__init__.py` | ✅ Created |
| `logic/src/models/embeddings/__init__.py` | ✅ Created |

### Phase 3: RL Pipeline ✅

| File | Status |
|------|--------|
| `logic/src/pipeline/rl/base.py` | ✅ Created |
| `logic/src/pipeline/rl/baselines.py` | ✅ Created |
| `logic/src/pipeline/rl/reinforce.py` | ✅ Created |
| `logic/src/pipeline/rl/__init__.py` | ✅ Created |
| `logic/src/pipeline/trainer.py` | ✅ Created |

### Phase 4: Data & Configuration ✅

| File | Status |
|------|--------|
| `logic/src/data/datasets.py` | ✅ Created |
| `logic/src/configs/__init__.py` | ✅ Created |

---

## Dependencies Added

```toml
# pyproject.toml
"tensordict>=0.3.0",
"pytorch-lightning>=2.0.0",
"hydra-core>=1.3.0",
"hydra-colorlog>=1.2.0",
"omegaconf>=2.3.0",
"lightning>=2.0.0",
```

---

## Key Architectural Changes

### 1. Environment Abstraction

**Before** (NamedTuple States):
```python
state = VRPP.make_state(input)
state = state.update(action)
```

**After** (RL4CO-style Environments):
```python
env = get_env("vrpp", generator=VRPPGenerator(num_loc=50))
td = env.reset(batch_size=256)
td["action"] = action
td = env.step(td)
```

### 2. TensorDict State Management

Unified state representation:
```python
td = TensorDict({
    "locs": Tensor,           # [batch, n_nodes, 2]
    "depot": Tensor,          # [batch, 2]
    "demand": Tensor,         # [batch, n_nodes]
    "current_node": Tensor,   # [batch]
    "visited": Tensor,        # [batch, n_nodes]
    "action_mask": Tensor,    # [batch, n_nodes]
    "done": Tensor,           # [batch]
}, batch_size=[batch_size])
```

### 3. PyTorch Lightning Training

```python
from logic.src.pipeline.rl.reinforce import REINFORCE
from logic.src.pipeline.trainer import WSTrainer

model = REINFORCE(env=env, policy=policy, baseline="rollout")
trainer = WSTrainer(max_epochs=100)
trainer.fit(model)
```

---

## Backward Compatibility

Old API will continue to work with deprecation warnings:

```python
# Old way (deprecated, will warn)
state = VRPP.make_state(input)

# New way (recommended)
env = get_env("vrpp")
td = env.reset(td)
```

Files deprecated (with warnings):
- `tasks/vrpp/state_vrpp.py`
- `tasks/vrpp/problem_vrpp.py`
- `tasks/wcvrp/state_wcvrp.py`
- `tasks/wcvrp/problem_wcvrp.py`

---

## Verification

```bash
# Test environment creation
python -c "from logic.src.envs import get_env; env = get_env('vrpp'); print(env)"

# Run existing test suite
python main.py test_suite --module test_problems
```

---

## References

- [RL4CO Repository](https://github.com/ai4co/rl4co)
- [TensorDict Documentation](https://pytorch.org/tensordict/)
- [PyTorch Lightning](https://lightning.ai/docs/pytorch/stable/)
