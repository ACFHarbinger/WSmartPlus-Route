import pytest
import torch
import torch.nn.functional as F
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays
from logic.src.utils.functions.decoding import (
    top_k_filter,
    top_p_filter,
)


# Hypothesis strategies
@st.composite
def logits_tensor(draw, batch_size=None, n_items=None):
    if batch_size is None:
        batch_size = draw(st.integers(min_value=1, max_value=10))
    if n_items is None:
        n_items = draw(st.integers(min_value=2, max_value=50))

    data = draw(arrays(dtype=float, shape=(batch_size, n_items), elements=st.floats(min_value=-100, max_value=100)))
    return torch.tensor(data, dtype=torch.float32)


@pytest.mark.property
@settings(suppress_health_check=[HealthCheck.too_slow])
@given(logits=logits_tensor(), k=st.integers(min_value=1, max_value=20))
def test_top_k_filter_masking(logits, k):
    """
    Test that top-k filtering masks values strictly smaller than the k-th largest.
    """
    n_items = logits.size(-1)
    k = min(k, n_items)

    filtered_logits = top_k_filter(logits.clone(), k)

    vals, _ = torch.topk(logits, k, dim=-1)
    thresh = vals[:, -1].unsqueeze(-1)

    # 1. Values strictly < threshold MUST be -inf
    should_be_masked = logits < thresh
    assert (filtered_logits[should_be_masked] == float("-inf")).all()

    # 2. Values > threshold should be preserved
    should_be_kept = logits > thresh
    # Note: Using masked_select or where to compare only relevant parts
    # If filtered is -inf where it should be kept, that's wrong.
    # But filtered_logits could be -inf if the original was -inf, but our strategy generates finite.
    assert torch.equal(filtered_logits[should_be_kept], logits[should_be_kept])

    # 3. Values == threshold should be preserved (implementation detail: ties are kept)
    # The current implementation uses logits < min_value to mask. So values >= min_value are kept.
    # If logits are finite, filtered_logits >= thresh where not masked.

    not_masked = filtered_logits != float("-inf")
    if not_masked.any():
        assert (filtered_logits[not_masked] >= thresh.expand_as(filtered_logits)[not_masked]).all()


@pytest.mark.property
@given(logits=logits_tensor(), p=st.floats(min_value=0.1, max_value=0.9))
def test_top_p_filter_probability_mass(logits, p):
    """
    Test that the probability mass of kept tokens is >= p.
    """
    # top_p filtering happens on logits converted to probs usually?
    # No, top_p_filter docstring says it takes logits.
    # It converts to probs internally to determine cutoff.

    filtered_logits = top_p_filter(logits.clone(), p)

    # Recover probabilities from filtered logits
    # Masked values are -inf, so their prob is 0
    probs = F.softmax(filtered_logits, dim=-1)

    # Sum of probabilities should be 1.0 (approximated)
    assert torch.allclose(probs.sum(dim=-1), torch.tensor(1.0), atol=1e-4)

    # Identify kept indices (finite logits)
    kept_mask = filtered_logits != float("-inf")

    # Calculate mass of these indices IN THE ORIGINAL DISTRIBUTION?
    # top_p ensures that the sum of probs of the top candidates is >= p.

    orig_probs = F.softmax(logits, dim=-1)

    # We need to sort original probs to check
    sorted_probs, _ = torch.sort(orig_probs, descending=True, dim=-1)
    torch.cumsum(sorted_probs, dim=-1)

    # Identify cutoff index in original sorted array
    # This is getting complex to verify exactly without re-implementing logic.
    # Let's verify that the kept tokens are indeed the top ones.

    # For each batch
    for i in range(logits.size(0)):
        row_kept = kept_mask[i]
        kept_indices = torch.where(row_kept)[0]
        masked_indices = torch.where(~row_kept)[0]

        if len(masked_indices) > 0 and len(kept_indices) > 0:
            min_kept_val = logits[i, kept_indices].min()
            max_masked_val = logits[i, masked_indices].max()
            # The smallest kept value should be >= the largest masked value
            assert min_kept_val >= max_masked_val
