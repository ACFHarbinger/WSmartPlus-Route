# Debugging Prompt

**Intent:** Resolve device placement issues using ReAct logic and existing project utilities.

## The Prompt

I am encountering a `RuntimeError: Expected all tensors to be on the same device` within the `AttentionModel` during training.

Context:
- I am running on an NVIDIA RTX 4080.
- The error seems to originate in `logic/src/models/modules/multi_head_attention.py`.

Task:
Analyze the provided code for `logic/src/models/attention_model.py` and `logic/src/models/modules/multi_head_attention.py`. Identify where input tensors might fail to be moved to the correct device.

Constraints:
- Do not suggest manual `.to('cuda')` calls. You MUST use the project's standard `logic/src/utils/setup_utils.py` or existing device properties in `logic/src/utils/definitions.py`.
- Check if the `ContextEmbedder` is correctly inheriting device placement.
- Provide the corrected code block and explain why this fixes the issue.
