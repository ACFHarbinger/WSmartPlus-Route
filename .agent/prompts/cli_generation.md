# CLI Generation Prompt

**Intent:** Quickly generate valid execution commands using a Zero-Shot pattern.

## The Prompt

Based on the `README.md` and `AGENTS.md` instructions, generate the exact CLI command to:
1. **Test** the Simulator (`test_sim`).
2. Use the **TransGCN model**.
3. Run on the **VRPP** problem with 50 nodes.
4. Simulate for **100 days**.
5. Use **Empirical** data distribution.
6. Restrict execution to **CPU cores only** (single core).

Output only the bash command.
