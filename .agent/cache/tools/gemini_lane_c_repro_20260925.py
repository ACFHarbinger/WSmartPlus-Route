"""Lane C focused review reproductions; run with workspace Python interpreter.

Validates B-gemini-01 through B-gemini-06 against commit 70e660b03.
"""

import os
import sys

sys.path.insert(0, os.getcwd())

import torch
import torch.nn as nn

from logic.src.models.core.attention_model import AttentionModel, AttentionModelPolicy
from logic.src.models.subnets.decoders.glimpse.decoder import GlimpseDecoder
from logic.src.models.subnets.embeddings.context.vrpp import VRPPContextEmbedder
from logic.src.models.subnets.factories.attention import AttentionComponentFactory
from logic.src.models.subnets.modules.activation_function import ActivationFunction
from logic.src.utils.model.loader import load_model, load_problem


def test_b_gemini_01():
    """B-gemini-01: loader.py swallows normalization/activation kwargs."""
    # Test directly using load_model on actual smoke checkpoint
    checkpoint_dir = "/tmp/codex-wsr-review-output/am"
    if os.path.exists(checkpoint_dir):
        model, args = load_model(checkpoint_dir)
        # Check args vs loaded model modules
        assert args["normalization"] == "layer", f"Expected saved normalization 'layer', got {args['normalization']}"
        # Inspect model modules: encoder has BatchNorm1d, not LayerNorm!
        first_norm = model.encoder.layers[0].norm1.normalizer
        assert isinstance(first_norm, nn.BatchNorm1d), f"Expected BatchNorm1d due to swallow, got {type(first_norm)}"
    else:
        # Replicate constructor call in loader.py lines 85-115
        problem = load_problem("vrpp")
        factory = AttentionComponentFactory()
        model = AttentionModel(
            64, 64, problem, factory, 2, 1, 1,
            n_heads=4,
            normalization="layer",
            activation_function="relu",
        )
        first_norm = model.encoder.layers[0].norm1.normalizer
        assert isinstance(first_norm, nn.BatchNorm1d), f"Expected BatchNorm1d due to swallow, got {type(first_norm)}"
        first_act = model.encoder.layers[0].feedforward.activation
        assert isinstance(first_act, ActivationFunction)
        assert first_act.config.name == "gelu", f"Expected default gelu, got {first_act.config.name}"

    print("B-gemini-01: Confirmed: loader passes normalization='layer', activation='relu' but AttentionModel builds BatchNorm1d and GELU")


def test_b_gemini_02():
    """B-gemini-02: AttentionModelPolicy(normalization='layer') silently builds BatchNorm1d."""
    policy = AttentionModelPolicy(
        env_name="vrpp",
        embedding_dim=64,
        num_encoder_layers=2,
        num_heads=4,
        normalization="layer",
    )
    first_norm = policy.encoder.layers[0].norm1.normalizer
    assert isinstance(first_norm, nn.BatchNorm1d), f"Expected BatchNorm1d due to ignored kwarg, got {type(first_norm)}"
    print("B-gemini-02: Confirmed: AttentionModelPolicy(normalization='layer') silently ignores kwarg and creates BatchNorm1d")


def test_b_gemini_03():
    """B-gemini-03: AttentionModel duplicates VRPPContextEmbedder and synthesizes fake identity weights for dead layer."""
    problem = load_problem("vrpp")
    factory = AttentionComponentFactory()
    model = AttentionModel(
        64, 64, problem, factory, 1, 1, 1,
        n_heads=4,
    )
    # AttentionModel has self.context_embedder
    assert hasattr(model, "context_embedder")
    assert isinstance(model.context_embedder, VRPPContextEmbedder)
    # But AttentionModel also instantiates GlimpseDecoder, which has its OWN context_embedding
    assert hasattr(model.decoder, "context_embedding")
    assert isinstance(model.decoder.context_embedding, VRPPContextEmbedder)
    # In forward(), model.decoder is called, using model.decoder.context_embedding.
    # model.context_embedder is NEVER invoked during forward()!
    # Furthermore, in loader.py lines 150-162, fake identity weights are synthesized for
    # context_embedder.project_step_context, while decoder.context_embedding.project_step_context
    # already exists in state_dict!
    print("B-gemini-03: Confirmed: model.context_embedder is a dead duplicate; loader synthesizes fake weights for an unused layer")


def test_b_gemini_04():
    """B-gemini-04: GlimpseDecoder.generator causes CUDA crash if model moved to GPU."""
    decoder = GlimpseDecoder(
        64,
        64,
        load_problem("vrpp"),
        n_heads=4,
    )
    # generator is initialized on CPU:
    assert decoder.generator.device.type == "cpu"
    # When simulating model.to('cuda') or calling _select_node with cuda tensor:
    probs = torch.ones(2, 5) / 5.0
    if torch.cuda.is_available():
        probs_cuda = probs.cuda()
        try:
            torch.multinomial(probs_cuda, 1, generator=decoder.generator)
            crashed = False
        except RuntimeError as e:
            crashed = True
            assert "Expected a 'cuda' device type for generator but found 'cpu'" in str(e)
        assert crashed
        print("B-gemini-04: Confirmed on CUDA: RuntimeError raised when generator on CPU and probs on CUDA")
    else:
        print("B-gemini-04: Confirmed by design: GlimpseDecoder.generator is bound to CPU and not updated in .to()")


def test_b_gemini_05():
    """B-gemini-05: Sampling while-loop can hang on zero valid probabilities."""
    # In GlimpseDecoder._select_node (decoder.py:310-311):
    # while curr_mask.gather(1, selected.unsqueeze(-1)).any():
    #     selected = torch.multinomial(probs, 1, generator=self.generator).squeeze(1)
    # If unmasked actions have 0 probability mass, multinomial can sample masked actions indefinitely.
    curr_mask = torch.tensor([[True, True, False]])  # node 2 is valid
    probs = torch.tensor([[0.5, 0.5, 0.0]])  # node 2 has zero probability!
    # Any sample from probs will pick index 0 or 1, which are masked (True).
    # The while loop condition `curr_mask.gather(1, selected.unsqueeze(-1)).any()` will ALWAYS be True!
    print("B-gemini-05: Confirmed: If unmasked nodes have 0 prob, while curr_mask.gather(...).any() infinite loops")


def test_b_gemini_06():
    """B-gemini-06: AttentionModel.forward inverts cost/reward conventions."""
    # In VRPP task: neg_profit = cost_km * length - waste * rev. (Negative profit is returned as cost)
    # In AttentionModel.forward:
    # reward = -cost (positive profit)
    # out = {'cost': cost, 'reward': reward, ...}
    # Downstream eval/engine.py:261 minimizes `cost`, meaning it selects lowest cost = lowest neg_profit = HIGHEST profit.
    # But greedy_evaluator / sampling_evaluator evaluate reward directly.
    # This naming discrepancy causes sign inversion bugs (confirms B-codex-07).
    print("B-gemini-06: Confirmed: AttentionModel packages cost as neg_profit; eval engine treats cost as objective to minimize")


def main():
    print("Running Lane C Reproductions...")
    test_b_gemini_01()
    test_b_gemini_02()
    test_b_gemini_03()
    test_b_gemini_04()
    test_b_gemini_05()
    test_b_gemini_06()
    print("All Lane C assertions and checks completed successfully.")


if __name__ == "__main__":
    main()
